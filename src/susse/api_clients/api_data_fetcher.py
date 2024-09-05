from abc import ABC


class ApiDataFetcher(ABC):
    """
    This class represents the base class for all other classes that fetch external data through an API (in particular
    all classes accessing the NASA satellite data
    """

    def __init__(self, base_url: str):
        self._base_url = base_url
