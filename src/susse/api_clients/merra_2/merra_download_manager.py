import requests
import logging
import yaml
from urllib import request, error
from http import cookiejar
from concurrent.futures import ThreadPoolExecutor
import re
import netCDF4 as nc
import numpy as np
from io import BytesIO
import os
import xarray as xr


class DownloadManager:
    __AUTHENTICATION_URL = 'https://urs.earthdata.nasa.gov/oauth/authorize'
    __username = ''
    __password = ''
    __download_url = ''
    __download_path = ''
    _authenticated_session = None

    def __init__(self, username='', password='', link=None, download_path='download'):
        self.set_username_and_password(username, password)
        self.download_url = link
        self.download_path = download_path

    @property
    def download_url(self):
        return self.__download_url

    @download_url.setter
    def download_url(self, link):
        """
        Setter for the links to download. The links have to be an array containing the URLs. The module will
        figure out the filename from the url and save it to the folder provided with download_path()
        :param links: The links to download
        :type links: List[str]
        """
        # TODO: Check if the links have the right structure?
        # Check if all links are formed properly
        if link is None:
            self.__download_url = ''
        else:
            try:
                self.get_filename(link[0])
            except AttributeError:
                raise ValueError(
                    'The URL seems to not have the right structure')
            self.__download_url = link

    @property
    def download_path(self):
        return self.__download_path

    @download_path.setter
    def download_path(self, file_path):
        self.__download_path = file_path
        return self.__download_path

    def set_username_and_password(self, username, password):
        self.__username = username
        self.__password = password

    def read_credentials_from_yaml(self):
        with open('Credentials.yml', 'r') as f:
            credentials = yaml.safe_load(f)
            Credentials = credentials['Credentials']
            self.set_username_and_password(
                Credentials['username'], Credentials['password'])
        return (print("Credentials Loaded"))

    def _mp_download_wrapper(self, url_item):
        """
        Wrapper for parallel download. Downloads a file from a URL and saves it.
        :param url_item: URL to download
        :type url_item: str
        """
        file_path = os.path.join(
            self.download_path, self.get_filename(url_item))
        self.__download_and_save_file(url_item, file_path)

    def start_download(self, nr_of_threads=4, save_file=False, product_name=None):
        if self._authenticated_session is None:
            self._authenticated_session = self.__create_authenticated_session()

        if save_file:
            os.makedirs(self.download_path, exist_ok=True)

            # Use threading or multiprocessing to download files in parallel

            with ThreadPoolExecutor(max_workers=nr_of_threads) as executor:
                executor.map(self._mp_download_wrapper, self.download_url)
        else:
            self.__download_and_load_file(product_name=product_name)

    @staticmethod
    def get_filename(url):
        """
        Extracts the filename from the URL.
        :param url: The MERRA-2 file URL
        :type url: str
        :return: Filename
        :rtype: str
        """
        reg_exp = r'(?<=/)[^/]*(?=.nc4?)'
        file_name = re.search(reg_exp, url).group(0)
        return file_name

    def __create_authenticated_session(self):
        """
        Creates an authenticated session using requests.
        :return: Authenticated session
        :rtype: requests.Session
        """
        s = requests.Session()
        s.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        s.auth = (self.__username, self.__password)
        s.cookies = self.__authorize_cookies_with_urllib()

        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            r = s.get(self.download_urls[0])
            logging.debug(f'Authentication Status: {r.status_code}')
            logging.debug(r.headers)
            logging.debug(r.cookies)

            logging.debug('Session Data')
            logging.debug(s.cookies)
            logging.debug(s.headers)
        return s

    def __authorize_cookies_with_urllib(self):
        """
        Authorizes the cookies needed to download files from the MERRA-2 server.
        :return: Cookies used for authentication
        :rtype: http.cookiejar.CookieJar
        """
        top_level_url = "https://urs.earthdata.nasa.gov"
        p = request.HTTPPasswordMgrWithDefaultRealm()
        p.add_password(None, top_level_url, self.__username, self.__password)

        auth_handler = request.HTTPBasicAuthHandler(p)
        auth_cookie_jar = cookiejar.CookieJar()
        cookie_jar = request.HTTPCookieProcessor(auth_cookie_jar)
        opener = request.build_opener(auth_handler, cookie_jar)

        request.install_opener(opener)

        try:
            # use fisrt url for authentication
            authentication_url = self.download_url[0]
            result = opener.open(authentication_url)
            logging.debug(f'Authentication successful: {result.status}')
            print(f'Authentication successful: {result.status}')
        except error.HTTPError as e:
            logging.error(f'HTTPError: {e.code} - {e.reason}')
            raise ValueError('Username and/or Password are incorrect!')
        except IndexError:
            raise IndexError('download_urls is not set')

        return auth_cookie_jar

    def __download_and_save_file(self, url, file_path):
        r = self._authenticated_session.get(url, stream=True)
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("File saved")
        return r.status_code

    def __download_and_load_file(self, product_name):
        """
        Downloads and loads the files into memory.
        :param urls: A single URL or a list of download URLs
        :param product_name: The product variable name to extract from the dataset
        :return: Data as a multidimensional numpy array
        """
        urls = self.download_url
        # Check if a single URL or a list of URLs is provided
        if isinstance(urls, str):
            # Convert to a list with one element for uniform handling
            urls = [urls]

        all_data = []

        # Iterate over each URL and download the corresponding file
        for url in urls:
            response = self._authenticated_session.get(url, stream=True)
            data = BytesIO(response.content)
            print("Loading File ..")

            # Load the dataset into memory and extract the desired variable data
            dataset = nc.Dataset('dummy', memory=data.read())
            array_data = np.array(dataset.variables[product_name][:])

            all_data.append(array_data)

        # Convert the list of arrays into a multidimensional numpy array
        # This will result in a (n, ...) shaped array, where n is the number of URLs
        return np.array(all_data)

    def xarrray_try(self):
        try:
            ds = xr.open_dataset(self.download_url[0])
            print(ds)
        except OSError as e:
            print('Error', e)
            print('Please Check your credentials')
