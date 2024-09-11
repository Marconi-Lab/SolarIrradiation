from abc import ABC


class ApiDataFetcher(ABC):
    """
    This class represents the base class for all other classes that fetch external data through an API (in particular
    all classes accessing the NASA satellite data
    """

    def __init__(self):
        pass
