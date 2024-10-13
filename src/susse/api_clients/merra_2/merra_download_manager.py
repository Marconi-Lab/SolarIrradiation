import getpass
import logging
import os
import re
import urllib.error
import urllib.request
import urllib.response
from http import cookiejar
from multiprocessing.dummy import Pool as Threadpool
from pathlib import Path
from typing import List, Tuple, Union
from urllib.parse import urlparse

import keyring
import keyring.errors
import requests

from susse.api_clients.merra_2.merra_product import MerraProductData

log = logging.getLogger("opendap_download")


class MerraDownloadManager:
    _TOP_LEVEL_URL = "https://urs.earthdata.nasa.gov"
    _SERVICE_NAME = "nasa_merra2"
    base_folder = Path(__file__).resolve().parents[4]
    _DOWNLOAD_FOLDER = str(base_folder / "merra_files_downloaded")

    def __init__(self):
        pass

    def get_credentials(self):
        try:
            # Attempt to retrieve credentials from the keyring
            username = keyring.get_password(
                MerraDownloadManager._SERVICE_NAME, "username"
            )
            password = keyring.get_password(
                MerraDownloadManager._SERVICE_NAME, "password"
            )

            # If credentials are not found, prompt the user and store them
            if username is None or password is None:
                print("NASA MERRA-2 credentials not found in keyring.")
                username = input("Enter your NASA-MERRA2 username: ")
                password = getpass.getpass("Enter your NASA-MERRA2 password: ")

                # Store credentials securely in the keyring
                keyring.set_password(
                    MerraDownloadManager._SERVICE_NAME, "username", username
                )
                keyring.set_password(
                    MerraDownloadManager._SERVICE_NAME, "password", password
                )
                print("Credentials stored securely in the keyring.")

            return username, password

        except keyring.errors.KeyringError as e:
            print(f"Keyring error: {e}")
            username = input("Enter your NASA username: ")
            password = getpass.getpass("Enter your NASA password: ")
            return username, password

    def download_from_urls(self, urls: Union[str, List[str]], nr_of_threads=4):
        p = Threadpool(nr_of_threads)
        if type(urls) is str:
            urls = [urls]
        p.map(self._mp_download_wrapper, urls)
        p.close()
        p.join()

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
        return file_name

    def get_folder_for_product(self, product: MerraProductData) -> str:
        return os.path.join(self._DOWNLOAD_FOLDER, product.product_name)

    def _mp_download_wrapper(self, url: str):
        """
        Wrapper for parallel download. The function name cannot start with __ due to visibility issues.
        """
        parsed_url = urlparse(url)
        query = parsed_url.query
        if query:
            # Get the substring before the first '['
            product_name = query.split("[")[0]
        else:
            logging.warning("Product name seems to be empty, url seems to be invalid!")
            product_name = ""

        product_folder = os.path.join(self._DOWNLOAD_FOLDER, product_name)
        if not os.path.exists(product_folder):
            os.makedirs(product_folder)
        file_name = MerraDownloadManager._extract_filename(url)
        file_path = os.path.join(product_folder, file_name)

        if os.path.exists(file_path):
            print(
                f"File '{file_name}' already exists in '{product_folder}'. Skipping download."
            )
        else:
            self.__download_and_save_file(url, file_path)

    def __download_and_save_file(self, url, file_path):
        authenticated_session = self.__create_authenticated_session(url)
        r = authenticated_session.get(url, stream=True)
        if r.status_code == 200:
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            return r.status_code
        else:
            logging.error(f"Failed to download {url}. Status code: {r.status_code}")
            logging.error(f"Response content: {r.text}")
            return r.status_code

    def __create_authenticated_session(self, url: str):
        s = requests.Session()
        s.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/40.0.2214.85 Safari/537.36"
        }
        user_name, pw = self.get_credentials()
        s.auth = (user_name, pw)
        s.cookies = self.__authorize_cookies_with_urllib(user_name, pw, url)

        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            r = s.get(url)
            log.debug("Authentication Status")
            log.debug(r.status_code)
            log.debug(r.headers)
            log.debug(r.cookies)

            log.debug("Sessions Data")
            log.debug(s.cookies)
            log.debug(s.headers)
        return s

    def __authorize_cookies_with_urllib(self, user_name: str, pw: str, url: str):

        # create an authorization handler
        p = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        p.add_password(None, self._TOP_LEVEL_URL, user_name, pw)

        auth_handler = urllib.request.HTTPBasicAuthHandler(p)
        auth_cookie_jar = cookiejar.CookieJar()
        cookie_jar = urllib.request.HTTPCookieProcessor(auth_cookie_jar)
        opener = urllib.request.build_opener(auth_handler, cookie_jar)

        urllib.request.install_opener(opener)

        try:
            # The merra portal moved the authentication to the download level. Before this change you had to
            # provide username and password on the overview page. For example:
            # goldsmr4.sci.gsfc.nasa.gov/opendap/MERRA2/M2T1NXSLV.5.12.4/
            # authentication_url = 'https://goldsmr4.sci.gsfc.nasa.gov/opendap/MERRA2/M2T1NXSLV.5.12.4/1980/01/MERRA2_100.tavg1_2d_slv_Nx.19800101.nc4.ascii?U2M[0:1:1][0:1:1][0:1:1]'
            # Changes:
            # Authenticate with the first url in the links.
            # Request the website and initialiaze the BasicAuth. This will populate the auth_cookie_jar
            result = opener.open(url)
            log.debug(list(auth_cookie_jar))
            log.debug(list(auth_cookie_jar)[0])
            log.debug(list(auth_cookie_jar)[1])

        except urllib.error.HTTPError as e:
            raise ValueError(f"Authorizing session failed due to HTTP error.{e}")
        except IOError as e:
            log.warning(e)
            raise IOError

        return auth_cookie_jar
