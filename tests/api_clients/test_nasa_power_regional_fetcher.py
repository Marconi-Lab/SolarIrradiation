"""Tests for the region-shaped NASA POWER fetcher.

The fetcher is a thin wrapper around an HTTP endpoint; tests focus on the
contracts that are easy to mis-implement and hard to debug from logs alone:

* URL construction matches the live endpoint's expected query schema.
* The bbox tiling respects the 4.5°-per-side cap with a safety margin.
* One HTTP call is made per (tile, api_code) — not per location.
* The FeatureCollection JSON parser produces the expected long shape and
  drops NASA POWER's -999 fill value.
"""

from __future__ import annotations

from datetime import date

import pytest

from susse.api_clients.NASA_Power import NASAPowerRegionalFetcher
from susse.api_clients.NASA_Power.nasa_power_regional_fetcher import _RegionalTile


# A bbox value object used by the tests. Inline-defined here so the
# tests don't pull in warehouse_ops (matches the production fetcher's
# primitive-arg signature).
class _Bbox:
    def __init__(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
    ) -> None:
        self.min_lat, self.max_lat = min_lat, max_lat
        self.min_lon, self.max_lon = min_lon, max_lon


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_bbox() -> _Bbox:
    """A bbox small enough to fit in a single regional request."""
    return _Bbox(min_lat=0.0, max_lat=2.0, min_lon=32.0, max_lon=34.0)


@pytest.fixture
def uganda_bbox() -> _Bbox:
    """The 6° × 5.55° Uganda bbox used by the a13 migration."""
    return _Bbox(min_lat=-1.5, max_lat=4.5, min_lon=29.5, max_lon=35.05)


@pytest.fixture
def short_dates() -> tuple[date, date]:
    return (date(2024, 1, 1), date(2024, 1, 3))


# ---------------------------------------------------------------------------
# Tile splitter
# ---------------------------------------------------------------------------


def _tile(bb: _Bbox) -> list:
    return NASAPowerRegionalFetcher._tile_bbox(
        bb.min_lat,
        bb.max_lat,
        bb.min_lon,
        bb.max_lon,
    )


class TestTileBbox:
    """Bbox → list[_RegionalTile]. The whole point of the splitter is to
    keep each tile inside NASA's 4.5° × 4.5° regional-endpoint cap."""

    def test_small_bbox_is_one_tile(self, small_bbox: _Bbox) -> None:
        tiles = _tile(small_bbox)
        assert len(tiles) == 1
        assert tiles[0].min_lat == small_bbox.min_lat
        assert tiles[0].max_lat == small_bbox.max_lat
        assert tiles[0].min_lon == small_bbox.min_lon
        assert tiles[0].max_lon == small_bbox.max_lon

    def test_uganda_bbox_splits_into_four_tiles(self, uganda_bbox: _Bbox) -> None:
        # 6° × 5.55° must split into a 2×2 grid because either span exceeds
        # the 4° safety margin (4.5° hard cap minus headroom).
        assert len(_tile(uganda_bbox)) == 4

    def test_every_tile_within_cap(self, uganda_bbox: _Bbox) -> None:
        # The whole point of tiling is to stay inside NASA's cap. If a tile
        # exceeds the documented 4.5° per side the endpoint returns 422.
        cap = NASAPowerRegionalFetcher._MAX_TILE_SPAN_DEG
        for tile in _tile(uganda_bbox):
            assert tile.span_lat <= cap + 1e-9
            assert tile.span_lon <= cap + 1e-9

    def test_tiles_cover_bbox_without_gaps(self, uganda_bbox: _Bbox) -> None:
        # Union of tile spans along each axis must equal the bbox span;
        # gaps would leave native pixels unfetched.
        tiles = _tile(uganda_bbox)
        min_lat = min(t.min_lat for t in tiles)
        max_lat = max(t.max_lat for t in tiles)
        min_lon = min(t.min_lon for t in tiles)
        max_lon = max(t.max_lon for t in tiles)
        assert min_lat == pytest.approx(uganda_bbox.min_lat)
        assert max_lat == pytest.approx(uganda_bbox.max_lat)
        assert min_lon == pytest.approx(uganda_bbox.min_lon)
        assert max_lon == pytest.approx(uganda_bbox.max_lon)


# ---------------------------------------------------------------------------
# URL builder
# ---------------------------------------------------------------------------


class TestBuildRegionalUrl:
    def test_url_uses_regional_path_and_query_schema(
        self, short_dates: tuple[date, date]
    ) -> None:
        tile = _RegionalTile(min_lat=0.0, max_lat=2.0, min_lon=32.0, max_lon=34.0)
        url = NASAPowerRegionalFetcher._build_regional_url(
            tile=tile,
            date_start=short_dates[0],
            date_end=short_dates[1],
            api_code="ALLSKY_SFC_SW_DWN",
        )
        assert "/temporal/daily/regional" in url
        assert "parameters=ALLSKY_SFC_SW_DWN" in url
        assert "latitude-min=0.0" in url and "latitude-max=2.0" in url
        assert "longitude-min=32.0" in url and "longitude-max=34.0" in url
        assert "start=20240101" in url
        assert "end=20240103" in url
        assert "community=RE" in url


# ---------------------------------------------------------------------------
# FeatureCollection parser
# ---------------------------------------------------------------------------


def _feature(lon: float, lat: float, api_code: str, daily: dict) -> dict:
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat, 1000.0]},
        "properties": {"parameter": {api_code: daily}},
    }


class TestParseFeatureCollection:
    """The parser is the bug-prone seam between NASA's response shape and
    the warehouse's long-format row schema. Pinning each invariant
    individually makes upstream schema drift loud."""

    def test_parses_multiple_features_into_long_rows(self) -> None:
        payload = {
            "features": [
                _feature(
                    32.5,
                    0.5,
                    "ALLSKY_SFC_SW_DWN",
                    {"20240101": 3.995, "20240102": 4.5},
                ),
                _feature(
                    33.5,
                    0.5,
                    "ALLSKY_SFC_SW_DWN",
                    {"20240101": 3.5198, "20240102": 5.5579},
                ),
            ],
        }
        rows = NASAPowerRegionalFetcher._parse_feature_collection(
            payload,
            api_code="ALLSKY_SFC_SW_DWN",
        )
        assert len(rows) == 4  # 2 features × 2 days

        first = rows[0]
        assert first["date"] == date(2024, 1, 1)
        assert first["latitude"] == 0.5
        assert first["longitude"] == 32.5
        assert first["variable_id"] == "ALLSKY_SFC_SW_DWN"
        assert first["value"] == pytest.approx(3.995)

    def test_drops_missing_value_fill(self) -> None:
        # NASA POWER signals missing data with -999. The warehouse must
        # never see this as a "real" value — long-format rows are
        # consumed downstream without further validation.
        payload = {
            "features": [
                _feature(
                    32.5,
                    0.5,
                    "T2M",
                    {"20240101": -999.0, "20240102": 25.5},
                ),
            ],
        }
        rows = NASAPowerRegionalFetcher._parse_feature_collection(
            payload,
            api_code="T2M",
        )
        assert len(rows) == 1
        assert rows[0]["date"] == date(2024, 1, 2)
        assert rows[0]["value"] == pytest.approx(25.5)

    def test_handles_empty_feature_collection(self) -> None:
        # An all-missing tile returns an empty FeatureCollection. The
        # parser must not crash on a missing `features` key either.
        assert (
            NASAPowerRegionalFetcher._parse_feature_collection(
                {},
                api_code="T2M",
            )
            == []
        )
        assert (
            NASAPowerRegionalFetcher._parse_feature_collection(
                {"features": []},
                api_code="T2M",
            )
            == []
        )


# ---------------------------------------------------------------------------
# fetch_region_long: end-to-end with a mocked HTTP layer
# ---------------------------------------------------------------------------


class TestFetchRegionLong:
    """End-to-end shape of the public API. Each test mocks the HTTP layer
    and verifies that the right URLs are called and the right DataFrame
    comes out."""

    def _call(
        self,
        bb: _Bbox,
        dates: tuple[date, date],
        api_codes: tuple[str, ...],
    ):
        return NASAPowerRegionalFetcher().fetch_region_long(
            min_lat=bb.min_lat,
            max_lat=bb.max_lat,
            min_lon=bb.min_lon,
            max_lon=bb.max_lon,
            date_start=dates[0],
            date_end=dates[1],
            api_codes=api_codes,
        )

    def test_one_call_per_tile_and_variable(
        self,
        requests_mock,
        uganda_bbox: _Bbox,
        short_dates: tuple[date, date],
    ) -> None:
        # Uganda bbox → 4 tiles. With 2 api_codes that's 8 HTTP calls.
        # Pin the count to protect the speedup; if a future refactor
        # re-introduces per-point looping the count balloons.
        requests_mock.get(
            "https://power.larc.nasa.gov/api/temporal/daily/regional",
            json={"features": []},
        )
        df = self._call(
            uganda_bbox,
            short_dates,
            ("ALLSKY_SFC_SW_DWN", "T2M"),
        )
        assert requests_mock.call_count == 4 * 2
        assert df.empty
        assert list(df.columns) == [
            "date",
            "latitude",
            "longitude",
            "variable_id",
            "value",
        ]

    def test_assembles_long_dataframe_across_tiles(
        self,
        requests_mock,
        small_bbox: _Bbox,
        short_dates: tuple[date, date],
    ) -> None:
        requests_mock.get(
            "https://power.larc.nasa.gov/api/temporal/daily/regional",
            json={
                "features": [
                    _feature(
                        32.5,
                        0.5,
                        "ALLSKY_SFC_SW_DWN",
                        {"20240101": 3.995, "20240102": 4.0, "20240103": 4.1},
                    ),
                    _feature(
                        33.5,
                        0.5,
                        "ALLSKY_SFC_SW_DWN",
                        {"20240101": 5.0, "20240102": 5.1, "20240103": 5.2},
                    ),
                ],
            },
        )
        df = self._call(small_bbox, short_dates, ("ALLSKY_SFC_SW_DWN",))
        assert len(df) == 6
        assert set(df["variable_id"]) == {"ALLSKY_SFC_SW_DWN"}
        assert set(df["latitude"]) == {0.5}
        assert set(df["longitude"]) == {32.5, 33.5}

    def test_empty_api_codes_returns_empty_frame_without_http(
        self,
        requests_mock,
        small_bbox: _Bbox,
        short_dates: tuple[date, date],
    ) -> None:
        # No work to do → no HTTP traffic, no surprise empty frames in
        # the wrong shape.
        df = self._call(small_bbox, short_dates, ())
        assert df.empty
        assert requests_mock.call_count == 0

    def test_rejects_inverted_bbox(
        self,
        short_dates: tuple[date, date],
    ) -> None:
        # min > max would silently fetch an empty rectangle on the live
        # endpoint; the fetcher must surface this as a clear ValueError
        # so the upstream caller can spot the swapped argument.
        inverted = _Bbox(min_lat=2.0, max_lat=0.0, min_lon=32.0, max_lon=34.0)
        with pytest.raises(ValueError, match="Invalid bbox"):
            self._call(inverted, short_dates, ("T2M",))
