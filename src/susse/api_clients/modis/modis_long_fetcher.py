"""Long-format MODIS fetcher for warehouse ingest.

Wraps the existing :class:`ModisDataFetcher` and adds the layer the
warehouse-population pipeline needs:

* Output is a long-format DataFrame with columns
  ``(date, product_id, band_id, value)`` — one row per
  (composite-end-date, product, band) measurement.
* **Per-product date-range chunking.** ORNL DAAC's RST subset API
  caps each request at 10 tiles, where a "tile" is one composite
  period — so a 16-day product accepts 160 days per request, an
  8-day product accepts 80, a daily product only 10. The fetcher
  walks the requested date range in product-aware chunks.
* **Concurrent fetches** across (product, location, chunk) work
  items via :class:`ThreadPoolExecutor`. ORNL DAAC is a public
  unauthenticated service and we keep concurrency modest.
* Errors on individual chunks are logged and skipped rather than
  aborting the whole batch.
"""

from __future__ import annotations

import logging
import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date as _date, datetime, timedelta

import pandas as pd

from .modis_data_fetcher import ModisDataFetcher
from .modis_data_result import ModisDataResult
from .modis_product import ModisProductEnum

_logger = logging.getLogger(__name__)

# Default thread-pool size for fan-out across (product, chunk) work
# items. ORNL DAAC is rarely the bottleneck on small per-station calls;
# bumping this past ~6 starts producing diminishing returns and is
# impolite to a public service. Override via constructor.
_DEFAULT_MAX_WORKERS = 4

# ORNL DAAC RST API caps each request at this many composite "tiles".
# Empirically observed (and confirmed by error message:
# "Subset time period ... exceeds maximum subset tiles support of 10").
_MAX_TILES_PER_REQUEST = 10

# Cadence in days per composite tile, by MODIS product. Used both to
# translate the 10-tile-per-request limit into a max-days-per-request
# budget and (downstream in warehouse-ingest jobs) to estimate how many
# rows a date range should produce when checking warehouse coverage.
# Add an entry here when introducing a new product to the catalog.
PRODUCT_CADENCE_DAYS: dict[str, int] = {
    "MCD43A4": 1,    # NBAR — daily 500 m
    "MOD11A2": 8,    # LST — 8-day composite
    "MOD13Q1": 16,   # NDVI — 16-day composite
}


def product_cadence_days(product_id: str) -> int:
    """Composite cadence in days for one MODIS product.

    Falls back to 1 (daily) with a warning for unknown products so callers
    don't crash on a typo, but this is also the wrong answer for non-daily
    products — register the product in :data:`PRODUCT_CADENCE_DAYS` to fix.
    """
    cadence = PRODUCT_CADENCE_DAYS.get(product_id)
    if cadence is None:
        _logger.warning(
            "Unknown MODIS product cadence for %r — assuming daily. Add it "
            "to PRODUCT_CADENCE_DAYS in %s for correct behaviour.",
            product_id, __name__,
        )
        cadence = 1
    return cadence


def _max_days_per_request(product_id: str) -> int:
    """Date-span ceiling for one ORNL DAAC subset request on a product."""
    return product_cadence_days(product_id) * _MAX_TILES_PER_REQUEST


def _date_chunks(
    start: _date, end: _date, max_days: int
) -> list[tuple[_date, _date]]:
    """Split ``[start, end]`` into contiguous spans of at most ``max_days``."""
    if start > end:
        return []
    chunks: list[tuple[_date, _date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=max_days - 1), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


class _PointLocation:
    """Minimal stand-in for ``geopy.location.Location``.

    :class:`ModisDataFetcher` reads only ``.latitude`` / ``.longitude``
    from the location object. Geopy's ``Location.__init__`` requires
    ``address`` and ``raw`` since 2.4, neither of which we have, so we
    use this slot-only adapter instead.
    """

    __slots__ = ("latitude", "longitude")

    def __init__(self, latitude: float, longitude: float) -> None:
        self.latitude = latitude
        self.longitude = longitude


class ModisLongFetcher:
    """Long-format MODIS fetcher.

    Constructor takes ``max_workers`` (default 4). Pass
    ``max_workers=1`` to disable parallelism — useful in tests where
    deterministic order matters.
    """

    def __init__(self, max_workers: int = _DEFAULT_MAX_WORKERS) -> None:
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}.")
        self._inner = ModisDataFetcher()
        self._max_workers = max_workers
        # Serialise the very first call so the inner fetcher's product-list
        # GET is hit only once even under concurrent kick-off.
        self._inner_lock = threading.Lock()
        self._inner_warmed = False

    def fetch_long_for_location(
        self,
        *,
        latitude: float,
        longitude: float,
        date_start: _date,
        date_end: _date,
        products_and_bands: tuple[tuple[str, str], ...],
    ) -> pd.DataFrame:
        """Fetch values for one location across a date range.

        Args:
            latitude: degrees, [-90, 90].
            longitude: degrees, [-180, 180].
            date_start: inclusive UTC date.
            date_end: inclusive UTC date.
            products_and_bands: tuples like
                ``(("MCD43A4", "Nadir_Reflectance_Band1"),
                  ("MOD11A2", "LST_Day_1km"))``.

        Returns:
            Long-format DataFrame with columns
            ``(date, product_id, band_id, value)``. Row order is
            non-deterministic when ``max_workers > 1``.
        """
        if date_start > date_end:
            raise ValueError(
                f"date_start ({date_start}) must be <= date_end ({date_end})."
            )
        if not products_and_bands:
            return pd.DataFrame(columns=("date", "product_id", "band_id", "value"))

        location = _PointLocation(latitude=latitude, longitude=longitude)

        # Build the unit-of-work list: one entry per (product, chunk).
        # Within a product, chunks are sized per its cadence so each
        # request stays under the 10-tile API limit.
        tasks: list[tuple[str, str, _date, _date]] = []
        for product_id, band_id in products_and_bands:
            chunks = _date_chunks(
                date_start, date_end, _max_days_per_request(product_id),
            )
            for chunk_start, chunk_end in chunks:
                tasks.append((product_id, band_id, chunk_start, chunk_end))

        if not tasks:
            return pd.DataFrame(columns=("date", "product_id", "band_id", "value"))

        # Warm up the inner fetcher's product-list cache once on the
        # main thread so worker threads don't race on first use.
        self._warm_up()

        if self._max_workers == 1:
            results = (self._fetch_one(*t, location) for t in tasks)
        else:
            pool = ThreadPoolExecutor(
                max_workers=self._max_workers, thread_name_prefix="modis-fetch"
            )
            futures = [pool.submit(self._fetch_one, *t, location) for t in tasks]
            results = (f.result() for f in as_completed(futures))

        try:
            rows: list[dict] = []
            for chunk in results:
                rows.extend(chunk)
        finally:
            if self._max_workers > 1:
                pool.shutdown(wait=True)

        return pd.DataFrame(
            rows, columns=("date", "product_id", "band_id", "value")
        )

    def _warm_up(self) -> None:
        with self._inner_lock:
            if self._inner_warmed:
                return
            # Touching the factory triggers the one-time products-list
            # GET. We don't need the result; just need it cached.
            try:
                self._inner._product_factory._fetch_products()  # noqa: SLF001
            except Exception:
                # If it fails here, individual fetches will surface the
                # error; don't block the whole batch.
                _logger.exception("Failed to warm up MODIS product list.")
            self._inner_warmed = True

    def _fetch_one(
        self,
        product_id: str,
        band_id: str,
        chunk_start: _date,
        chunk_end: _date,
        location: _PointLocation,
    ) -> list[dict]:
        try:
            product_enum = ModisProductEnum(product_id)
        except ValueError:
            _logger.warning(
                "Skipping unknown MODIS product_id=%r (no matching "
                "ModisProductEnum member). Add it to modis_product.py if "
                "you intend to use it.",
                product_id,
            )
            return []
        start_dt = datetime.combine(chunk_start, datetime.min.time())
        end_dt = datetime.combine(chunk_end, datetime.min.time())
        _logger.info(
            "MODIS request: product=%s band=%s lat=%.4f lon=%.4f %s..%s",
            product_id, band_id, location.latitude, location.longitude,
            chunk_start.isoformat(), chunk_end.isoformat(),
        )
        t0 = _time.monotonic()
        try:
            result: ModisDataResult | None = self._inner.fetch_product_result(
                product_enum=product_enum,
                location=location,
                start_date=start_dt,
                end_date=end_dt,
                band_name=band_id,
            )
        except Exception as exc:
            _logger.warning(
                "MODIS fetch failed for product=%s band=%s %s..%s: %s",
                product_id, band_id, chunk_start, chunk_end, exc,
            )
            return []
        elapsed = _time.monotonic() - t0
        if result is None or not result.data_points:
            return []
        rows = self._result_to_rows(result, product_id, band_id)
        _logger.info(
            "MODIS response: product=%s band=%s — %d rows in %.1fs.",
            product_id, band_id, len(rows), elapsed,
        )
        return rows

    @staticmethod
    def _result_to_rows(
        result: ModisDataResult, product_id: str, band_id: str
    ) -> list[dict]:
        rows: list[dict] = []
        for dp in result.data_points:
            value = float(dp.data_avg)
            if pd.isna(value):
                continue
            rows.append({
                "date": dp.date.date(),
                "product_id": product_id,
                "band_id": band_id,
                "value": value,
            })
        return rows
