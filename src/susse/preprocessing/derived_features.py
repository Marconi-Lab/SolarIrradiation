"""Pure helper functions for derived features.

Stateless, deterministic transformations of input columns into model
features. Each function takes one or more pandas series and returns
either a series (single new feature) or a DataFrame (related new
features). No fitting, no internal state — the same inputs always
produce the same outputs.

Keeping these as free functions (rather than methods on the
:class:`Preprocessor`) makes them straightforward to unit-test against
analytical references and lets notebooks call them directly when
exploring features outside the formal pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Below this denominator, kt is considered ill-defined and we emit NaN
# rather than dividing through. The night-time / nadir-zenith case is
# the obvious source — any clear-sky-derived denominator near zero is
# numerically unstable and physically meaningless for an ML feature.
_KT_DENOMINATOR_FLOOR: float = 1e-6


def clear_sky_index(
    ghi: pd.Series,
    ghi_clear: pd.Series,
    *,
    denominator_floor: float = _KT_DENOMINATOR_FLOOR,
) -> pd.Series:
    """Return ``kt = GHI / GHI_clear``, with NaN for ill-defined denominators.

    ``kt`` is the canonical "how cloudy is it" feature: 1.0 means the
    sky is at clear-sky reference intensity, < 1 means attenuation,
    occasionally > 1 from cloud-edge enhancement. It collapses two
    quantities into one normalised signal that's much easier for a
    learner than raw GHI alone (which mixes diurnal / seasonal cycles
    with cloud-state information).

    Args:
        ghi: Series of GHI values (any consistent unit; common is
            kWh/m²/day for the daily warehouse).
        ghi_clear: Series of clear-sky reference values, same unit.
        denominator_floor: Values of ``ghi_clear`` below this are
            treated as ill-defined (rather than producing huge spikes
            from division). Default is :data:`_KT_DENOMINATOR_FLOOR`.

    Returns:
        Series of the same index as the inputs.

    Raises:
        ValueError: If ``ghi`` and ``ghi_clear`` have mismatched indices.
    """
    if not ghi.index.equals(ghi_clear.index):
        raise ValueError(
            "clear_sky_index expects ghi and ghi_clear to share an index. "
            "Pass aligned series — re-index with .reindex(...) at the call "
            "site if needed."
        )
    safe_denom = ghi_clear.where(ghi_clear.abs() >= denominator_floor)
    return (ghi / safe_denom).rename("kt")


def cyclical_day_of_year(dates: pd.Series) -> pd.DataFrame:
    """Sin/cos encoding of day-of-year for a yearly cyclical feature.

    Returns two columns ``doy_sin``, ``doy_cos`` so that adjacent
    calendar dates (e.g. Dec 31 and Jan 1) sit close in feature space
    rather than at opposite ends of a 1..365 ramp. Standard treatment
    for any periodic time-of-year feature; the model can recover the
    raw day-of-year via ``atan2(sin, cos)`` if needed.

    Args:
        dates: Series of ``datetime`` / ``date`` values.

    Returns:
        DataFrame with two columns sharing the input index.

    Raises:
        ValueError: If ``dates`` is empty.
    """
    if len(dates) == 0:
        raise ValueError("cyclical_day_of_year received an empty Series.")
    doy = pd.to_datetime(dates).dt.dayofyear.astype(float)
    # Use 366 (max possible day count) rather than 365 so leap-year days
    # don't fall outside [0, 1) and the encoding stays well-defined for
    # every calendar date.
    angle = 2.0 * np.pi * (doy - 1.0) / 366.0
    return pd.DataFrame(
        {"doy_sin": np.sin(angle), "doy_cos": np.cos(angle)},
        index=dates.index,
    )
