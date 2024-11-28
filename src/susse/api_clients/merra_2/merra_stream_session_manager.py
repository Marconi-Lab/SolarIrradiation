from pydap.cas.urs import setup_session

from .merra_download_manager import MerraDownloadManager


class StreamSessionManager:
    """
    Manages session authentication for streaming MERRA-2 data.

    This class is responsible for handling user credentials and authenticating
    sessions using the MerraDownloadManager.

    """

    def __init__(self):
        self._authenticate_session = MerraDownloadManager()

    def authenticate(self, url: str) -> None:
        """
        Authenticates the session .
        """
        try:
            user_name, pw = self._authenticate_session.get_credentials()
            session = setup_session(user_name, pw, check_url=url)
            return session
        except Exception as e:
            raise Exception(f"Failed to authenticate session : {e}")
