"""Tests for :mod:`susse.warehouse_ops.snapping`.

The snapper is the single boundary that turns an arbitrary query coordinate
into a warehouse ``geohash5``. The contracts that matter:

* a query point snaps to the genuinely-nearest stored cell, and
* a point outside the ingested footprint is reported as out of coverage
  (``None`` / :class:`OutOfCoverageError`) rather than snapped to a far,
  unrelated cell.
"""

from __future__ import annotations

import pandas as pd
import pygeohash
import pytest

from susse.warehouse_ops.snapping import NearestPixelSnapper, OutOfCoverageError

# A patch of NASA POWER's MERRA-2 native grid (0.5° lat × 0.625° lon) over
# Uganda. Real grid geometry — the snapper carries no hard-coded grid, so
# this doubles as both fixture and the thing under test.
_GRID_LATS = [0.0, 0.5, 1.0]
_GRID_LONS = [32.5, 33.125, 33.75]


def _grid_pixels() -> tuple[list[str], list[float], list[float]]:
    lats: list[float] = []
    lons: list[float] = []
    ghs: list[str] = []
    for lat in _GRID_LATS:
        for lon in _GRID_LONS:
            lats.append(lat)
            lons.append(lon)
            ghs.append(pygeohash.encode(lat, lon, precision=5))
    return ghs, lats, lons


@pytest.fixture
def grid_snapper() -> NearestPixelSnapper:
    ghs, lats, lons = _grid_pixels()
    return NearestPixelSnapper(geohash5s=ghs, latitudes=lats, longitudes=lons)


class TestSnap:
    def test_exact_pixel_snaps_to_itself(
        self, grid_snapper: NearestPixelSnapper
    ) -> None:
        # A coordinate sitting exactly on a pixel centre must return that
        # pixel's geohash5 — the degenerate, zero-distance case.
        expected = pygeohash.encode(0.5, 33.125, precision=5)
        assert grid_snapper.snap(0.5, 33.125) == expected

    def test_off_grid_point_snaps_to_nearest_centre(
        self, grid_snapper: NearestPixelSnapper
    ) -> None:
        # (0.3, 32.6) is closest to the (0.5, 32.5) pixel centre, not to the
        # (0.0, 32.5) one below it nor the (0.5, 33.125) one to its east.
        assert grid_snapper.snap(0.3, 32.6) == pygeohash.encode(0.5, 32.5, precision=5)

    def test_out_of_coverage_point_raises(
        self, grid_snapper: NearestPixelSnapper
    ) -> None:
        # A point thousands of km from every pixel must not snap silently.
        with pytest.raises(OutOfCoverageError, match="outside the source's"):
            grid_snapper.snap(40.0, 10.0)


class TestSnapOrNone:
    def test_batch_matches_per_point_snap(
        self, grid_snapper: NearestPixelSnapper
    ) -> None:
        lats = [0.0, 0.3, 0.9]
        lons = [32.5, 32.6, 33.7]
        batch = grid_snapper.snap_or_none(lats, lons)
        per_point = [grid_snapper.snap(la, lo) for la, lo in zip(lats, lons)]
        assert batch == per_point

    def test_empty_batch_returns_empty(self, grid_snapper: NearestPixelSnapper) -> None:
        assert grid_snapper.snap_or_none([], []) == []

    def test_mismatched_lengths_raise(self, grid_snapper: NearestPixelSnapper) -> None:
        with pytest.raises(ValueError, match="must align"):
            grid_snapper.snap_or_none([0.0, 0.5], [32.5])

    def test_out_of_coverage_points_become_none(
        self, grid_snapper: NearestPixelSnapper
    ) -> None:
        # In-coverage points snap; far points yield None — the batch keeps
        # input order so a caller can tell exactly which points missed.
        result = grid_snapper.snap_or_none([0.3, 40.0, 0.9], [32.6, 10.0, 33.7])
        assert result[0] == pygeohash.encode(0.5, 32.5, precision=5)
        assert result[1] is None
        assert result[2] == pygeohash.encode(1.0, 33.75, precision=5)


class TestConstruction:
    def test_mismatched_input_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="differ in length"):
            NearestPixelSnapper(
                geohash5s=["a", "b"], latitudes=[0.0], longitudes=[32.5]
            )

    def test_empty_pixel_set_reports_no_coverage(self) -> None:
        # A source with no warehouse rows yields a snapper that misses every
        # point — an explicit cache-miss signal, never a wrong snap.
        snapper = NearestPixelSnapper(geohash5s=[], latitudes=[], longitudes=[])
        assert snapper.n_pixels == 0
        assert snapper.snap_or_none([0.3, 1.0], [32.6, 33.0]) == [None, None]
        with pytest.raises(OutOfCoverageError):
            snapper.snap(0.3, 32.6)

    def test_out_of_range_latitude_raises(self) -> None:
        with pytest.raises(ValueError, match=r"latitudes.*\[-90, 90\]"):
            NearestPixelSnapper(geohash5s=["a"], latitudes=[123.0], longitudes=[32.5])

    def test_non_positive_guard_raises(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            NearestPixelSnapper(
                geohash5s=["a"], latitudes=[0.0], longitudes=[32.5], max_snap_km=0.0
            )


class TestFromDataFrame:
    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame({"geohash5": ["a"], "latitude": [0.0]})  # no longitude
        with pytest.raises(ValueError, match="missing column"):
            NearestPixelSnapper.from_dataframe(df)

    def test_round_trips_through_dataframe(self) -> None:
        ghs, lats, lons = _grid_pixels()
        df = pd.DataFrame({"geohash5": ghs, "latitude": lats, "longitude": lons})
        snapper = NearestPixelSnapper.from_dataframe(df)
        assert snapper.n_pixels == len(ghs)
        assert snapper.snap(0.3, 32.6) == pygeohash.encode(0.5, 32.5, precision=5)


def test_guard_distance_admits_in_grid_query() -> None:
    # The default 120 km guard must comfortably admit a point at the worst
    # case for a 1° (CERES) grid: the dead centre of a cell, ~78 km from the
    # four surrounding pixel centres.
    ghs = [pygeohash.encode(la, lo, 5) for la in (0.5, 1.5) for lo in (32.5, 33.5)]
    lats = [la for la in (0.5, 1.5) for _ in (32.5, 33.5)]
    lons = [lo for _ in (0.5, 1.5) for lo in (32.5, 33.5)]
    snapper = NearestPixelSnapper(geohash5s=ghs, latitudes=lats, longitudes=lons)
    assert snapper.snap(1.0, 33.0) in ghs
