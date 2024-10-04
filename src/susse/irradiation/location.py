from dataclasses import dataclass

import pvlib


@dataclass
class Location(pvlib.location.Location):
    """Location class for storing location information using pvlib.Location

    Location objects are convenient containers for latitude, longitude,
    timezone, and altitude data associated with a particular
    geographic location. You can also assign a name to a location object.

    Parameters
    ----------
    latitude : float
        Positive values are north of the equator.
        Use decimal degrees notation (e.g. 14.23).

    longitude : float
        Positive values are east of the prime meridian.
        Use decimal degrees notation (e.g. -170.34).

    timezone : str
        Timezone name from the IANA Time Zone Database.
        See pvlib.iotools.read_tz_world for a list of possible values.

    altitude : float, default None
        Altitude above sea level in meters.
        If None, the altitude will be looked up using pvlib.location.lookup_altitude.

    **kwargs
        Additional keyword arguments to be passed to the pvlib.Location constructor.

    See Also
    --------
    pvlib.location.Location
    """

    def __init__(
        self,
        latitude: float,
        longitude: float,
        timezone: str,
        altitude: float = None,
        **kwargs,
    ):
        if altitude is None:
            altitude = pvlib.location.lookup_altitude(latitude, longitude)

        super().__init__(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            tz=timezone,
            **kwargs,
        )
