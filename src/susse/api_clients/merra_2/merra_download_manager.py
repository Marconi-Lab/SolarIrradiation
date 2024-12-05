import getpass
import logging
import os
import re
import urllib.error
import urllib.request
import urllib.response
from functools import partial
from http import cookiejar
from multiprocessing.dummy import Pool as Threadpool
from typing import List, Optional, Tuple, Union

import keyring
import keyring.errors
import requests
from cryptography.fernet import Fernet
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class MerraDownloadManager:
    """
    This class handles the download of MERRA2 files. It also takes care of the authentification and storage of username
    and passwords. The basic logic of authentification is based on the code available here;
    https://github.com/emilylaiken/merradownload

    The user credentials are stored in a local keyring, but can also be set from outside through set_username_pw.
    Inside the class the credentials are stored only in encrypted form

    To obtain a username and password one has to register https://urs.earthdata.nasa.gov/
    """

    _TOP_LEVEL_URL = "https://urs.earthdata.nasa.gov"
    _SERVICE_NAME = "nasa_merra2"
    _KEYRING_ENCRYPTION_KEY = "fernet_key"

    def __init__(self) -> None:
        self._auth_session: Optional[requests.Session] = None
        self._encrypted_username: Optional[bytes] = None
        self._encrypted_password: Optional[bytes] = None
        self.cipher_suite = self._get_or_create_cipher_key()

    def _get_or_create_cipher_key(self) -> Fernet:
        """
        Fetch or generate an encryption key for Fernet and store it in the keyring.
        """
        key_str = keyring.get_password(
            MerraDownloadManager._SERVICE_NAME, self._KEYRING_ENCRYPTION_KEY
        )
        if key_str is None:
            # Generate a new key and store it in the keyring
            key = Fernet.generate_key()
            keyring.set_password(
                MerraDownloadManager._SERVICE_NAME,
                self._KEYRING_ENCRYPTION_KEY,
                key.decode(),
            )
            logging.info("Generated and stored encryption key securely.")
        else:
            key = (
                key_str.encode()
            )  # The keyring returns it as a string, convert back to bytes

        return Fernet(key)

    def set_username_pw(self, username: str, password: str) -> None:
        """
        Encrypts and stores the username and password.
        """
        self._encrypted_username = self.cipher_suite.encrypt(username.encode())
        self._encrypted_password = self.cipher_suite.encrypt(password.encode())

    def get_credentials(self) -> Tuple[str, str]:
        if (
            self._encrypted_username is not None
            and self._encrypted_password is not None
        ):
            decrypted_username = self.cipher_suite.decrypt(
                self._encrypted_username
            ).decode()
            decrypted_password = self.cipher_suite.decrypt(
                self._encrypted_password
            ).decode()
            return decrypted_username, decrypted_password

        try:
            username = keyring.get_password(
                MerraDownloadManager._SERVICE_NAME, "username"
            )
            password = keyring.get_password(
                MerraDownloadManager._SERVICE_NAME, "password"
            )

            if username is None or password is None:
                logging.info(
                    "NASA GES-DISC for MERRA-2 credentials not found in keyring."
                )
                username = input("Enter your NASA GES-DISC for MERRA-2 username: ")
                password = getpass.getpass(
                    "Enter your NASA GES-DISC for MERRA-2 password: "
                )

                keyring.set_password(
                    MerraDownloadManager._SERVICE_NAME, "username", username
                )
                keyring.set_password(
                    MerraDownloadManager._SERVICE_NAME, "password", password
                )
                logging.info("Credentials stored securely in the keyring.")

            return username, password

        except keyring.errors.KeyringError as e:
            logging.error(f"Keyring error: {e}")
            username = input("Enter your NASA GES-DISC username for MERRA-2: ")
            password = getpass.getpass(
                "Enter your NASA GES-DISC password for MERRA-2: "
            )
            self.set_username_pw(username, password)
            return username, password

    def session_authenticated(self) -> bool:
        return self._auth_session is not None

    def authenticate_session(self, authentication_url: str) -> None:
        self._auth_session = self.__create_authenticated_session(authentication_url)
        if self._auth_session:
            logging.info("Session authenticated successfully.")
        else:
            logging.error("Failed to authenticate session.")

    def download_from_urls(
        self, urls: Union[str, List[str]], download_folder: str, nr_of_threads=4
    ):
        if type(urls) is str:
            urls = [urls]

        if not os.path.exists(download_folder):
            os.makedirs(download_folder)

        download_wrapper_with_folder = partial(
            self._mp_download_wrapper, download_folder=download_folder
        )
        with Threadpool(nr_of_threads) as p:
            p.map(download_wrapper_with_folder, urls)

    @staticmethod
    def extract_filename_from_url(url: str) -> str:
        """
        Extracts the filename from the url. This method can also be used to check
        if the links have the correct structure

        """
        # Extract everything between a leading / and .nc4? . The problem with using this without any
        # other classification is, that the URLs have multiple / in their structure. The expressions [^/]* matches
        # everything but /. Combined with the outer expressions, this only matches the part between the last / and .nc4?
        reg_exp = r"(?<=/)[^/]*(?=.nc4?)"
        matched_entries = re.search(reg_exp, url)
        file_name = matched_entries.group(0) if matched_entries else ""
        if file_name:
            logging.info(f"Extracted filename: {file_name}")
        else:
            logging.warning(f"Failed to extract filename from URL: {url}")
        return file_name

    def _mp_download_wrapper(self, url: str, download_folder: str):
        """
        Wrapper for parallel download. The function name cannot start with __ due to visibility issues.
        """
        file_name = MerraDownloadManager.extract_filename_from_url(url)
        file_path = os.path.join(download_folder, file_name)

        if os.path.exists(file_path):
            logging.info(
                f"File '{file_name}' already exists in '{download_folder}'. Skipping download."
            )
        else:
            self.__download_and_save_file(url, file_path)

    def __download_and_save_file(self, url: str, file_path: str) -> None:
        if not self.session_authenticated():
            self.authenticate_session(url)
            if not self.session_authenticated():
                logging.error("Failed to authenticate session!")
                return

        assert self._auth_session is not None

        try:
            r = self._auth_session.get(url, stream=True)
            if r.status_code == 200:
                with open(file_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                logging.info(f"Downloaded and saved file: {file_path}")
            else:
                logging.error(
                    f"Failed to download {url}. Status code: {r.status_code}\n"
                )
                logging.error(f"Response content: {r.text}\n")
        except requests.RequestException as e:
            logging.error(f"Error during download: {e}")
            return

    def __create_authenticated_session(
        self,
        download_url: str,
    ) -> Optional[requests.Session]:
        """
        The merra portal seems to behave rather difficult when it comes to authentication. It seems that you need to
        set the cookies manually to make sure that the authentication is saved. I do not fully understand why, but
        I did not manage to find a more simple way
        :param download_url: a url to a downloadable file
        :return: requests.Session that corresponds to an authenticated session
        """
        try:
            session = requests.Session()

            retry = Retry(connect=3, backoff_factor=0.5)
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("https://", adapter)

            # The session headers simulate a web-browser
            session.headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/40.0.2214.85 Safari/537.36"
            }

            user_name, pw = self.get_credentials()
            session.auth = (user_name, pw)
            session.cookies = self.__get_authentication_cookies(download_url)

            r = session.get(download_url)

            if r.status_code == 200:
                logging.info("Authenticated successfully.")
                return session
            else:
                logging.error(
                    f"Authentication failed with status code: {r.status_code}"
                )
                logging.error(f"Response content: {r.text}")
                return None

        except Exception as e:
            logging.error(f"Failed to create authenticated session: {e}")
            return None

    def __get_authentication_cookies(
        self, url: str
    ) -> requests.cookies.RequestsCookieJar:
        try:
            user_name, pw = self.get_credentials()

            # Create an authorization handler for basic HTTP authentication
            p = urllib.request.HTTPPasswordMgrWithDefaultRealm()
            p.add_password(None, self._TOP_LEVEL_URL, user_name, pw)

            auth_handler = urllib.request.HTTPBasicAuthHandler(p)
            auth_cookie_jar = cookiejar.CookieJar()
            cookie_jar = urllib.request.HTTPCookieProcessor(auth_cookie_jar)
            opener = urllib.request.build_opener(auth_handler, cookie_jar)

            urllib.request.install_opener(opener)

            # Open the URL to authenticate and get the cookies
            # The merra portal moved the authentication to the download level.
            opener.open(url)

            logging.info("Cookies successfully retrieved.")

            # Convert cookies from cookiejar.CookieJar to requests.cookies.RequestsCookieJar
            requests_cookie_jar = requests.cookies.RequestsCookieJar()
            for cookie in auth_cookie_jar:
                if cookie.value is not None:
                    requests_cookie_jar.set(
                        cookie.name,
                        cookie.value,
                        domain=cookie.domain,
                        path=cookie.path,
                    )

            return requests_cookie_jar

        except urllib.error.HTTPError as e:
            logging.error(f"HTTP error during cookie retrieval: {e}")
            raise ValueError("Failed to authorize due to HTTP error")
        except Exception as e:
            logging.error(f"Error during cookie retrieval: {e}")
            raise
