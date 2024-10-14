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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class MerraDownloadManager:
    _TOP_LEVEL_URL = "https://urs.earthdata.nasa.gov"
    _SERVICE_NAME = "nasa_merra2"

    def __init__(self) -> None:
        self._auth_session: Optional[requests.Session] = None

    @staticmethod
    def get_credentials() -> Tuple[str, str]:
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

                # Store credentials securely in the keyring
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
    def _extract_filename(url: str) -> str:
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
        file_name = MerraDownloadManager._extract_filename(url)
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
        self, download_url: str
    ) -> Optional[requests.Session]:
        try:
            s = requests.Session()

            retry = Retry(connect=3, backoff_factor=0.5)
            adapter = HTTPAdapter(max_retries=retry)
            s.mount("https://", adapter)

            s.headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/40.0.2214.85 Safari/537.36"
            }

            user_name, pw = MerraDownloadManager.get_credentials()
            s.auth = (user_name, pw)
            s.cookies = self.__get_right_cookies(download_url)

            r = s.get(download_url)

            if r.status_code == 200:
                logging.info("Authenticated successfully.")
                return s
            else:
                logging.error(
                    f"Authentication failed with status code: {r.status_code}"
                )
                logging.error(f"Response content: {r.text}")
                return None

        except Exception as e:
            logging.error(f"Failed to create authenticated session: {e}")
            return None

    def __get_right_cookies(self, url: str) -> requests.cookies.RequestsCookieJar:
        try:
            user_name, pw = MerraDownloadManager.get_credentials()

            # Create an authorization handler for basic HTTP authentication
            p = urllib.request.HTTPPasswordMgrWithDefaultRealm()
            p.add_password(None, self._TOP_LEVEL_URL, user_name, pw)

            auth_handler = urllib.request.HTTPBasicAuthHandler(p)
            auth_cookie_jar = cookiejar.CookieJar()
            cookie_jar = urllib.request.HTTPCookieProcessor(auth_cookie_jar)
            opener = urllib.request.build_opener(auth_handler, cookie_jar)

            urllib.request.install_opener(opener)

            # Open the URL to authenticate and get the cookies
            # The merra portal moved the authentication to the download level. Before this change you had to
            # provide username and password on the overview page. For example:
            # goldsmr4.sci.gsfc.nasa.gov/opendap/MERRA2/M2T1NXSLV.5.12.4/
            # authentication_url = 'https://goldsmr4.sci.gsfc.nasa.gov/opendap/MERRA2/M2T1NXSLV.5.12.4/1980/01/MERRA2_100.tavg1_2d_slv_Nx.19800101.nc4.ascii?U2M[0:1:1][0:1:1][0:1:1]'
            opener.open(url)

            logging.info("Cookies successfully retrieved.")

            # Convert cookies from cookiejar.CookieJar to requests.cookies.RequestsCookieJar
            requests_cookie_jar = requests.cookies.RequestsCookieJar()
            for cookie in auth_cookie_jar:
                requests_cookie_jar.set(
                    cookie.name, cookie.value, domain=cookie.domain, path=cookie.path
                )

            return requests_cookie_jar

        except urllib.error.HTTPError as e:
            logging.error(f"HTTP error during cookie retrieval: {e}")
            raise ValueError("Failed to authorize due to HTTP error")
        except Exception as e:
            logging.error(f"Error during cookie retrieval: {e}")
            raise
