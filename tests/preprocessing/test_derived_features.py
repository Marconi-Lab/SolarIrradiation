"""Tests for the pure derived-feature helpers."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from susse.preprocessing import clear_sky_index, cyclical_day_of_year


class TestClearSkyIndex:
    """``kt = ghi / ghi_clear`` with NaN for ill-defined denominators."""

    def test_typical_values_match_division(self) -> None:
        ghi = pd.Series([3.0, 4.0, 5.0])
        ghi_clear = pd.Series([6.0, 5.0, 5.0])
        kt = clear_sky_index(ghi, ghi_clear)
        np.testing.assert_allclose(kt.values, [0.5, 0.8, 1.0])

    def test_zero_denominator_yields_nan(self) -> None:
        # GHI_clear of 0 is night-time / nadir-zenith — kt is undefined.
        # Must return NaN rather than +inf or arbitrary fill.
        ghi = pd.Series([3.0, 4.0])
        ghi_clear = pd.Series([0.0, 5.0])
        kt = clear_sky_index(ghi, ghi_clear)
        assert pd.isna(kt.iloc[0])
        assert kt.iloc[1] == pytest.approx(0.8)

    def test_below_floor_denominator_yields_nan(self) -> None:
        # Even a tiny positive denominator (1e-10) would produce huge kt
        # values that aren't physically meaningful. The floor catches it.
        ghi = pd.Series([3.0])
        ghi_clear = pd.Series([1e-10])
        assert pd.isna(clear_sky_index(ghi, ghi_clear).iloc[0])

    def test_mismatched_indices_raise(self) -> None:
        ghi = pd.Series([1.0, 2.0], index=[0, 1])
        ghi_clear = pd.Series([3.0, 4.0], index=[10, 11])
        with pytest.raises(ValueError, match="aligned"):
            clear_sky_index(ghi, ghi_clear)

    def test_negative_clear_sky_treated_as_floor_violation(self) -> None:
        # A clear-sky reference that's slightly negative is data-quality
        # corruption. Treat it as undefined rather than producing a sign-
        # flipped kt.
        ghi = pd.Series([3.0])
        ghi_clear = pd.Series([-1e-10])  # |x| < floor
        assert pd.isna(clear_sky_index(ghi, ghi_clear).iloc[0])


class TestCyclicalDayOfYear:
    """sin/cos encoding: continuous across year boundary, period 1 year."""

    def test_january_first_lands_at_zero_angle(self) -> None:
        # Day 1 maps to angle 0 → sin=0, cos=1.
        out = cyclical_day_of_year(pd.Series([date(2024, 1, 1)]))
        assert out["doy_sin"].iloc[0] == pytest.approx(0.0, abs=1e-9)
        assert out["doy_cos"].iloc[0] == pytest.approx(1.0, abs=1e-9)

    def test_dec31_close_to_jan1_in_feature_space(self) -> None:
        # The whole point of the cyclical encoding: adjacent calendar
        # days near the year boundary are close in feature space, not
        # at the opposite ends of a 1..365 ramp.
        out = cyclical_day_of_year(
            pd.Series([date(2024, 12, 31), date(2025, 1, 1)])
        )
        # L2 distance in (sin, cos) space.
        d = np.hypot(
            out["doy_sin"].iloc[0] - out["doy_sin"].iloc[1],
            out["doy_cos"].iloc[0] - out["doy_cos"].iloc[1],
        )
        # Adjacent days are < ~2π/366 apart on the unit circle, so the
        # chord should be small (< 0.05).
        assert d < 0.05, f"Dec 31 and Jan 1 should be near-adjacent; got d={d}"

    def test_summer_and_winter_solstices_are_opposite(self) -> None:
        # Days ~6 months apart should be roughly opposite on the unit
        # circle.
        out = cyclical_day_of_year(
            pd.Series([date(2024, 1, 1), date(2024, 7, 2)])  # day 1 vs ~day 184
        )
        d = np.hypot(
            out["doy_sin"].iloc[0] - out["doy_sin"].iloc[1],
            out["doy_cos"].iloc[0] - out["doy_cos"].iloc[1],
        )
        # Opposite points on the unit circle have chord 2.0.
        assert d > 1.9, f"~6-month-apart days should be ~opposite; got d={d}"

    def test_leap_year_day_366_still_well_defined(self) -> None:
        # 2024 is a leap year. Dec 31 is day 366. With a /366 normaliser,
        # the encoding stays well-defined and close to day 365 of a
        # non-leap year (no wrap-around glitch).
        out = cyclical_day_of_year(pd.Series([date(2024, 12, 31)]))
        assert np.isfinite(out["doy_sin"].iloc[0])
        assert np.isfinite(out["doy_cos"].iloc[0])

    def test_empty_series_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            cyclical_day_of_year(pd.Series([], dtype="datetime64[ns]"))

    def test_index_preserved(self) -> None:
        # Caller may have a non-default index; the output must align so
        # downstream merges don't silently re-index.
        idx = [10, 20, 30]
        s = pd.Series(
            [date(2024, 1, 1), date(2024, 4, 1), date(2024, 7, 1)],
            index=idx,
        )
        out = cyclical_day_of_year(s)
        assert list(out.index) == idx
