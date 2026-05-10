"""Tests for DerivedFeature subclasses + the kind-dispatch JSON loader.

Per CLAUDE.md, these test the public contract of each concrete
:class:`DerivedFeature` — what it produces, what inputs it needs, and
that JSON roundtrip preserves the spec (modulo provider injection).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from susse.preprocessing import (
    AltitudeFeature,
    ClearSkyIndexFeature,
    CyclicalDayOfYearFeature,
    DerivedFeature,
    FeatureKind,
    derived_feature_from_dict,
)


def _frame_with_coords() -> pd.DataFrame:
    return pd.DataFrame({
        "date": [date(2024, 1, 1), date(2024, 1, 2)],
        "lat": [0.5179, 0.5179],
        "lon": [32.4715, 32.4715],
        "ghi": [4.5, 5.5],
        "ghi_clear": [6.0, 6.0],
    })


class TestClearSkyIndexFeature:
    def test_compute_matches_division(self) -> None:
        f = ClearSkyIndexFeature(
            ghi_column="ghi", ghi_clear_column="ghi_clear",
            output_column="kt",
        )
        result = f.compute(_frame_with_coords())
        assert list(result.columns) == ["kt"]
        np.testing.assert_allclose(result["kt"].values, [4.5 / 6.0, 5.5 / 6.0])

    def test_required_input_columns_match_construction(self) -> None:
        f = ClearSkyIndexFeature(
            ghi_column="my_ghi", ghi_clear_column="my_clear",
            output_column="kt",
        )
        assert f.required_input_columns == ("my_ghi", "my_clear")
        assert f.output_columns == ("kt",)

    def test_empty_field_rejected(self) -> None:
        with pytest.raises(ValueError, match="output_column"):
            ClearSkyIndexFeature(
                ghi_column="g", ghi_clear_column="gc", output_column="",
            )

    def test_dict_roundtrip(self) -> None:
        f = ClearSkyIndexFeature(
            ghi_column="g", ghi_clear_column="gc", output_column="kt",
        )
        d = f.to_dict()
        assert d["kind"] == FeatureKind.CLEAR_SKY_INDEX.value
        assert derived_feature_from_dict(d) == f


class TestCyclicalDayOfYearFeature:
    def test_compute_emits_two_columns(self) -> None:
        f = CyclicalDayOfYearFeature()
        result = f.compute(_frame_with_coords())
        assert list(result.columns) == ["doy_sin", "doy_cos"]

    def test_no_fields(self) -> None:
        # Output column names are fixed by design — no per-instance config.
        a = CyclicalDayOfYearFeature()
        b = CyclicalDayOfYearFeature()
        assert a == b
        assert a.output_columns == ("doy_sin", "doy_cos")
        assert a.required_input_columns == ("date",)

    def test_dict_roundtrip(self) -> None:
        f = CyclicalDayOfYearFeature()
        d = f.to_dict()
        assert d == {"kind": FeatureKind.CYCLICAL_DOY.value}
        assert derived_feature_from_dict(d) == f


class _StubProvider:
    def __init__(self, value: float = 1500.0) -> None:
        self._value = value
        self.calls: list[tuple[float, float]] = []

    def __call__(self, lat: float, lon: float) -> float:
        self.calls.append((lat, lon))
        return self._value


class TestAltitudeFeature:
    def test_compute_emits_altitude_column(self) -> None:
        provider = _StubProvider(value=1200.0)
        f = AltitudeFeature(provider=provider)
        result = f.compute(_frame_with_coords())
        assert list(result.columns) == ["altitude_m"]
        assert (result["altitude_m"] == 1200.0).all()

    def test_dedupes_provider_calls_per_unique_coord(self) -> None:
        provider = _StubProvider()
        f = AltitudeFeature(provider=provider)
        # Both rows in the test frame share the same (lat, lon) — one call.
        f.compute(_frame_with_coords())
        assert len(provider.calls) == 1

    def test_required_input_columns_are_lat_lon(self) -> None:
        f = AltitudeFeature(provider=_StubProvider())
        assert f.required_input_columns == ("lat", "lon")

    def test_dict_drops_provider(self) -> None:
        # Provider isn't JSON-serialisable; to_dict must omit it.
        f = AltitudeFeature(provider=_StubProvider())
        d = f.to_dict()
        assert "provider" not in d
        assert d == {
            "kind": FeatureKind.ALTITUDE.value,
            "output_column": "altitude_m",
        }

    def test_from_dict_without_provider_raises(self) -> None:
        # Re-injection is mandatory and the error message tells the
        # caller exactly which dict key to pass.
        d = {
            "kind": FeatureKind.ALTITUDE.value,
            "output_column": "altitude_m",
        }
        with pytest.raises(ValueError, match="altitude"):
            derived_feature_from_dict(d)

    def test_from_dict_with_provider_reconstructs(self) -> None:
        provider = _StubProvider()
        d = {
            "kind": FeatureKind.ALTITUDE.value,
            "output_column": "altitude_m",
        }
        f = derived_feature_from_dict(d, providers={"altitude": provider})
        assert isinstance(f, AltitudeFeature)
        assert f.provider is provider


class TestDispatchByKind:
    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(ValueError):
            derived_feature_from_dict({"kind": "not_a_real_kind"})

    def test_missing_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            derived_feature_from_dict({"output_column": "kt"})


class TestProtocolCompliance:
    """Every concrete DerivedFeature must implement the abstract contract."""

    def test_all_subclasses_expose_kind_and_output(self) -> None:
        instances: list[DerivedFeature] = [
            ClearSkyIndexFeature(
                ghi_column="g", ghi_clear_column="gc", output_column="kt",
            ),
            CyclicalDayOfYearFeature(),
            AltitudeFeature(provider=_StubProvider()),
        ]
        for f in instances:
            assert isinstance(f.kind, FeatureKind)
            assert len(f.output_columns) >= 1
            assert len(f.required_input_columns) >= 1
            assert "kind" in f.to_dict()
