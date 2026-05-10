"""Tests for FeatureSpec — validation + JSON roundtrip on the derived_features API."""

from __future__ import annotations

import pytest

from susse.preprocessing import (
    AltitudeFeature,
    ClearSkyIndexFeature,
    CyclicalDayOfYearFeature,
    FeatureSpec,
)


def _kt(out: str = "kt_nasa") -> ClearSkyIndexFeature:
    return ClearSkyIndexFeature(
        ghi_column="sat_ghi_nasa_kwh_m2_day",
        ghi_clear_column="nasa_ghi_clear",
        output_column=out,
    )


class _StubProvider:
    """Stand-in for a real ElevationProvider in tests that don't run compute()."""

    def __call__(self, lat: float, lon: float) -> float:
        return 0.0


class TestValidation:
    def test_empty_target_rejected(self) -> None:
        with pytest.raises(ValueError, match="target_column"):
            FeatureSpec(target_column="")

    def test_duplicate_derived_outputs_rejected(self) -> None:
        # Two kt features producing the same output column would
        # silently clobber each other downstream — surface at construction.
        with pytest.raises(ValueError, match="duplicate"):
            FeatureSpec(derived_features=(_kt("kt"), _kt("kt")))

    def test_derived_output_colliding_with_feature_rejected(self) -> None:
        # A derived output that's also a pass-through feature would be
        # doubly-defined. Validation lives on FeatureSpec, not on the
        # individual feature, because the collision is cross-cutting.
        with pytest.raises(ValueError, match="collide with feature_columns"):
            FeatureSpec(
                feature_columns=("kt_nasa",),
                derived_features=(_kt("kt_nasa"),),
            )

    def test_doy_columns_collision_with_features_rejected(self) -> None:
        # CyclicalDayOfYearFeature outputs ('doy_sin', 'doy_cos'); a
        # pass-through with the same name is the same kind of collision.
        with pytest.raises(ValueError, match="doy_sin"):
            FeatureSpec(
                feature_columns=("doy_sin",),
                derived_features=(CyclicalDayOfYearFeature(),),
            )

    def test_altitude_collision_with_feature_rejected(self) -> None:
        with pytest.raises(ValueError, match="altitude_m"):
            FeatureSpec(
                feature_columns=("altitude_m",),
                derived_features=(AltitudeFeature(provider=_StubProvider()),),
            )


class TestOutputFeatureNames:
    """Order: pass-through first, then each DerivedFeature's output_columns."""

    def test_passthrough_then_derived_in_order(self) -> None:
        spec = FeatureSpec(
            feature_columns=("nasa_aod_550", "nasa_precipitable_water"),
            derived_features=(
                _kt("kt_nasa"),
                CyclicalDayOfYearFeature(),
                AltitudeFeature(provider=_StubProvider()),
            ),
        )
        assert spec.output_feature_names == (
            "nasa_aod_550",
            "nasa_precipitable_water",
            "kt_nasa",
            "doy_sin",
            "doy_cos",
            "altitude_m",
        )

    def test_minimal_spec_emits_nothing(self) -> None:
        spec = FeatureSpec()
        assert spec.output_feature_names == ()

    def test_doy_only_emits_two_columns(self) -> None:
        spec = FeatureSpec(derived_features=(CyclicalDayOfYearFeature(),))
        assert spec.output_feature_names == ("doy_sin", "doy_cos")


class TestJsonRoundtrip:
    """Spec → JSON → spec; provider re-injection at load time."""

    def test_kt_and_doy_roundtrip_without_providers(self) -> None:
        spec = FeatureSpec(
            target_column="y_ghi_kwh_m2_day",
            feature_columns=("nasa_aod_550",),
            derived_features=(_kt("kt_nasa"), CyclicalDayOfYearFeature()),
            id_columns=("date", "location"),
        )
        roundtripped = FeatureSpec.from_json(spec.to_json())
        assert roundtripped == spec

    def test_minimal_spec_roundtrip(self) -> None:
        spec = FeatureSpec()
        roundtripped = FeatureSpec.from_json(spec.to_json())
        assert roundtripped == spec

    def test_altitude_roundtrip_requires_provider_injection(self) -> None:
        provider = _StubProvider()
        spec = FeatureSpec(
            derived_features=(AltitudeFeature(provider=provider),),
        )
        s = spec.to_json()
        # Without providers, the from_json must raise — the spec
        # carries an AltitudeFeature whose provider was dropped at
        # serialisation, and silently constructing it with a placeholder
        # would mask the configuration error.
        with pytest.raises(ValueError, match="altitude"):
            FeatureSpec.from_json(s)
        # With the provider re-injected, the roundtrip succeeds.
        roundtripped = FeatureSpec.from_json(s, providers={"altitude": provider})
        assert roundtripped == spec
