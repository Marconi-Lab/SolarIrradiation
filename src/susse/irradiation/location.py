from dataclasses import dataclass
from typing import Optional

import pvlib


@dataclass
class Location(pvlib.location.Location):
    def __init__(
        self, latitude: float, longitude: float, timezone: str, altitude: float = None
    ):
        if altitude is None:
            altitude = pvlib.location.lookup_altitude(latitude, longitude)
        super().__init__(
            latitude=latitude, longitude=longitude, altitude=altitude, tz=timezone
        )
