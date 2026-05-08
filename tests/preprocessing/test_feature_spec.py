"""Tests for FeatureSpec — validation + JSON roundtrip."""

from __future__ import annotations

import pytest

from susse.preprocessing import ClearSkyIndexSpec, FeatureSpec


def _kt_spec(out: str = "kt_nasa") -> ClearSkyIndexSpec:
    return ClearSkyIndexSpec(
        ghi_column="sat_ghi_nasa_kwh_m2_day",
        ghi_clear_column="nasa_ghi_clear",
        output_column=out,
    )


class TestValidation:
    def test_empty_target_rejected(self) -> None:
        with pytest.raises(ValueError, match="target_column"):
            FeatureSpec(target_column="")

    def test_duplicate_kt_outputs_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            FeatureSpec(
                clear_sky_index_specs=(
                    _kt_spec("kt"),
                    _kt_spec("kt"),  # collision
                ),
            )

    def test_kt_output_colliding_with_feature_rejected(self) -> None:
        # A kt output that's also listed as a pass-through feature would
        # be doubly-defined (raw column AND derived column with the same
        # name). Surface that at construction.
        with pytest.raises(ValueError, match="collides with feature_columns"):
            FeatureSpec(
                feature_columns=("kt_nasa",),
                clear_sky_index_specs=(_kt_spec("kt_nasa"),),
            )

    def test_doy_columns_collision_with_features_rejected(self) -> None:
        with pytest.raises(ValueError, match="doy_sin"):
            FeatureSpec(
                feature_columns=("doy_sin",),
                include_cyclical_doy=True,
            )


class TestOutputFeatureNames:
    """Ordered list of model-input columns the preprocessor will produce."""

    def test_pass_through_then_kt_then_cyclical(self) -> None:
        spec = FeatureSpec(
            feature_columns=("nasa_aod_550", "nasa_precipitable_water"),
            clear_sky_index_specs=(_kt_spec("kt_nasa"),),
            include_cyclical_doy=True,
        )
        assert spec.output_feature_names == (
            "nasa_aod_550",
            "nasa_precipitable_water",
            "kt_nasa",
            "doy_sin",
            "doy_cos",
        )

    def test_minimal_spec_emits_only_doy(self) -> None:
        spec = FeatureSpec(include_cyclical_doy=True)
        assert spec.output_feature_names == ("doy_sin", "doy_cos")

    def test_no_features_no_doy_yields_empty(self) -> None:
        spec = FeatureSpec(include_cyclical_doy=False)
        assert spec.output_feature_names == ()


class TestJsonRoundtrip:
    def test_full_spec_roundtrip(self) -> None:
        spec = FeatureSpec(
            target_column="y_ghi_kwh_m2_day",
            feature_columns=("nasa_aod_550",),
            clear_sky_index_specs=(_kt_spec("kt_nasa"),),
            include_cyclical_doy=True,
            id_columns=("date", "location"),
            dropna_target=True,
            dropna_features=True,
        )
        roundtripped = FeatureSpec.from_json(spec.to_json())
        assert roundtripped == spec

    def test_minimal_spec_roundtrip(self) -> None:
        spec = FeatureSpec()
        roundtripped = FeatureSpec.from_json(spec.to_json())
        assert roundtripped == spec
