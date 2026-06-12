"""Tests for FeatureService's point-based snap / fetch / relabel path.

FeatureService takes plain (lat, lon) query points, snaps each onto every
source's native grid independently, and relabels the fetched rows back to
the query point's own geohash5. These tests pin that contract with the
SatelliteRepository methods stubbed — no BigQuery round-trip.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pygeohash
import pytest

from susse.datasets import FeatureSelection, FeatureService
from susse.warehouse_ops.io.repositories import SatelliteRepository
from susse.warehouse_ops.population.types import IrradianceBand, Source

# Two NASA native pixels (MERRA-2-spaced) over Uganda. The warehouse stores
# data at these centres; query points anywhere near them must snap here.
_PIXEL_A = (0.5, 32.5)
_PIXEL_B = (0.5, 33.0)
_PIXEL_A_GH = pygeohash.encode(*_PIXEL_A, precision=5)
_PIXEL_B_GH = pygeohash.encode(*_PIXEL_B, precision=5)


@pytest.fixture
def nasa_only_selection() -> FeatureSelection:
    return FeatureSelection(
        nasa_variable_ids=("temperature",),
        cams_variable_ids=(),
        include_satellite_irradiance=(Source.NASA_POWER,),
        include_satellite_bands=(IrradianceBand.GHI,),
        qc_levels=("pass",),
    )


class _FakeBQ:
    """Only `config` is touched — every query goes through the stubbed repo."""

    @property
    def config(self):
        from susse.warehouse_ops.io.config import WarehouseConfig

        return WarehouseConfig()


@pytest.fixture
def stub_repo(monkeypatch: pytest.MonkeyPatch):
    """Stub SatelliteRepository: two NASA native pixels, one day of data each."""

    def fake_native_pixels(self, *, table_fqn, source=None):
        return pd.DataFrame(
            {
                "geohash5": [_PIXEL_A_GH, _PIXEL_B_GH],
                "latitude": [_PIXEL_A[0], _PIXEL_B[0]],
                "longitude": [_PIXEL_A[1], _PIXEL_B[1]],
            }
        )

    def fake_irradiance(self, start, end, *, sources, bands, geohash5s):
        # One row per requested pixel; GHI value encodes which pixel it is
        # so the relabel can be checked (4.0 for A, 5.0 for B).
        rows = []
        for gh in geohash5s:
            rows.append(
                {
                    "date": date(2024, 1, 1),
                    "geohash5": gh,
                    "sat_ghi_nasa_kwh_m2_day": 4.0 if gh == _PIXEL_A_GH else 5.0,
                }
            )
        return pd.DataFrame(rows)

    def fake_aux(
        self, *, table_fqn, column_prefix, start, end, variable_ids, geohash5s
    ):
        rows = [
            {"date": date(2024, 1, 1), "geohash5": gh, "nasa_temperature": 25.0}
            for gh in geohash5s
        ]
        return pd.DataFrame(rows)

    monkeypatch.setattr(SatelliteRepository, "native_pixels", fake_native_pixels)
    monkeypatch.setattr(
        SatelliteRepository, "daily_irradiance_by_geohash", fake_irradiance
    )
    monkeypatch.setattr(SatelliteRepository, "long_aux_pivoted", fake_aux)


def _service() -> FeatureService:
    return FeatureService(_FakeBQ())  # type: ignore[arg-type]


class TestSnapAndRelabel:
    def test_query_point_snaps_to_pixel_but_keys_by_query_geohash(
        self, stub_repo, nasa_only_selection: FeatureSelection
    ) -> None:
        # A point ~3 km off pixel A: it must read pixel A's data (GHI 4.0)
        # but the output row must be keyed by the *query* point's geohash5,
        # not the pixel's.
        query_pt = (0.52, 32.52)
        df = _service().build_satellite_features(
            selection=nasa_only_selection,
            date_start=date(2024, 1, 1),
            date_end=date(2024, 1, 1),
            points=[query_pt],
        )
        assert len(df) == 1
        assert df.iloc[0]["geohash5"] == pygeohash.encode(*query_pt, precision=5)
        assert df.iloc[0]["sat_ghi_nasa_kwh_m2_day"] == 4.0
        assert df.iloc[0]["nasa_temperature"] == 25.0

    def test_two_points_in_one_pixel_each_get_the_pixel_data(
        self, stub_repo, nasa_only_selection: FeatureSelection
    ) -> None:
        # Two query points ~6 km apart — distinct geohash5 cells, same NASA
        # pixel A. Both must appear, each under its own geohash5, both
        # carrying pixel A's value.
        pt_1 = (0.50, 32.50)
        pt_2 = (0.54, 32.54)
        df = _service().build_satellite_features(
            selection=nasa_only_selection,
            date_start=date(2024, 1, 1),
            date_end=date(2024, 1, 1),
            points=[pt_1, pt_2],
        )
        assert set(df["geohash5"]) == {
            pygeohash.encode(*pt_1, precision=5),
            pygeohash.encode(*pt_2, precision=5),
        }
        assert set(df["sat_ghi_nasa_kwh_m2_day"]) == {4.0}

    def test_distinct_pixels_resolve_independently(
        self, stub_repo, nasa_only_selection: FeatureSelection
    ) -> None:
        # One point near pixel A, one near pixel B → 4.0 and 5.0 respectively.
        df = _service().build_satellite_features(
            selection=nasa_only_selection,
            date_start=date(2024, 1, 1),
            date_end=date(2024, 1, 1),
            points=[(0.51, 32.51), (0.49, 32.99)],
        )
        assert sorted(df["sat_ghi_nasa_kwh_m2_day"]) == [4.0, 5.0]

    def test_out_of_coverage_point_yields_no_row(
        self, stub_repo, nasa_only_selection: FeatureSelection
    ) -> None:
        # A point on another continent has no NASA pixel within the snap
        # guard → no irradiance spine → no row (a downstream cache miss).
        df = _service().build_satellite_features(
            selection=nasa_only_selection,
            date_start=date(2024, 1, 1),
            date_end=date(2024, 1, 1),
            points=[(48.0, 2.0)],
        )
        assert df.empty

    def test_empty_points_raises(
        self, nasa_only_selection: FeatureSelection
    ) -> None:
        with pytest.raises(ValueError, match="points is empty"):
            _service().build_satellite_features(
                selection=nasa_only_selection,
                date_start=date(2024, 1, 1),
                date_end=date(2024, 1, 1),
                points=[],
            )
