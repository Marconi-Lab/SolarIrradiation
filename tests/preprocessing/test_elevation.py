"""Tests for the elevation provider — caching, Protocol compliance, fail-loud.

Tests do NOT hit the live Open-Elevation endpoint. We monkeypatch
``pvlib.location.lookup_altitude`` so the test suite stays offline and
deterministic; the contract being verified is the provider's caching +
error-propagation behaviour, not pvlib's network code.
"""

from __future__ import annotations

import pytest

from susse.preprocessing import ElevationProvider, PvlibElevationProvider


class TestPvlibElevationProvider:
    """The production provider's contract: cache hits, fail-loud, single source."""

    def test_repeated_lookup_uses_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[float, float]] = []

        def fake_lookup(*, latitude: float, longitude: float) -> float:
            calls.append((latitude, longitude))
            return 1234.5

        monkeypatch.setattr("pvlib.location.lookup_altitude", fake_lookup)
        provider = PvlibElevationProvider()
        a = provider(0.5179, 32.4715)
        b = provider(0.5179, 32.4715)
        assert a == b == 1234.5
        assert len(calls) == 1
        assert provider.cache_size == 1

    def test_distinct_points_each_hit_pvlib_once(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[tuple[float, float]] = []

        def fake_lookup(*, latitude: float, longitude: float) -> float:
            calls.append((latitude, longitude))
            return 100.0 * len(calls)

        monkeypatch.setattr("pvlib.location.lookup_altitude", fake_lookup)
        provider = PvlibElevationProvider()
        provider(0.0, 32.0)
        provider(1.0, 33.0)
        provider(0.0, 32.0)  # cached
        assert len(calls) == 2
        assert provider.cache_size == 2

    def test_subdecimal_coordinates_round_to_same_cache_key(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 4-decimal rounding (~10 m at the equator) is below any free DEM's
        # spatial resolution, so two near-identical lookups must coalesce
        # to one network call rather than redundantly hitting the API.
        calls: list[tuple[float, float]] = []

        def fake_lookup(*, latitude: float, longitude: float) -> float:
            calls.append((latitude, longitude))
            return 500.0

        monkeypatch.setattr("pvlib.location.lookup_altitude", fake_lookup)
        provider = PvlibElevationProvider()
        provider(0.51790000, 32.47150000)
        provider(0.51790001, 32.47150001)  # within rounding
        assert len(calls) == 1

    def test_pvlib_failure_propagates(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # No silent fallback to NaN / 0 — a misconfigured network must
        # surface as a clear error rather than poison a feature column.
        def fake_lookup(*, latitude: float, longitude: float) -> float:
            raise RuntimeError("simulated network failure")

        monkeypatch.setattr("pvlib.location.lookup_altitude", fake_lookup)
        provider = PvlibElevationProvider()
        with pytest.raises(RuntimeError, match="simulated network failure"):
            provider(0.5, 32.5)


class TestProtocolCompliance:
    """Any callable matching the Protocol must work with the Preprocessor."""

    def test_minimal_callable_satisfies_protocol(self) -> None:
        # Smallest possible implementation: a lambda. Verifies the
        # Protocol does not require anything beyond the call signature.
        provider: ElevationProvider = lambda lat, lon: 0.0  # noqa: E731
        assert provider(0.0, 0.0) == 0.0
