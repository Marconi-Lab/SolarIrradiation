import pytest

from susse.irradiation import Location


def test_location_initialization():
    kampala = Location(latitude=0.347594, longitude=32.58210, timezone="Africa/Kampala")

    assert kampala.latitude == pytest.approx(0.347594)
    assert kampala.longitude == pytest.approx(32.58210)
    assert kampala.tz == "Africa/Kampala"
    assert kampala.altitude is not None  # Altitude should be automatically looked up


def test_location_with_custom_altitude():
    custom_altitude = 1202.0
    kampala = Location(
        latitude=0.347594,
        longitude=32.58210,
        timezone="Africa/Kampala",
        altitude=custom_altitude,
    )

    assert kampala.altitude == custom_altitude


def test_location_attributes():
    kampala = Location(latitude=0.347594, longitude=32.58210, timezone="Africa/Kampala")

    assert hasattr(kampala, "latitude")
    assert hasattr(kampala, "longitude")
    assert hasattr(kampala, "altitude")
    assert hasattr(kampala, "tz")


def test_invalid_location():
    with pytest.raises(IndexError):
        Location(
            latitude=91, longitude=32.58210, timezone="Africa/Kampala"
        )  # Invalid latitude

    with pytest.raises(IndexError):
        Location(
            latitude=0.347594, longitude=181, timezone="Africa/Kampala"
        )  # Invalid longitude
