from multiprocessing.dummy import Pool as Threadpool
import requests
import logging
import yaml
from urllib import request, error
from http import cookiejar
import re
import netCDF4 as nc
import numpy as np
from io import BytesIO
import os


class DownloadManager:
    __AUTHENTICATION_URL = 'https://urs.earthdata.nasa.gov/oauth/authorize'
    __username = ''
    __password = ''
    __download_urls = []
    _authenticated_session = None

    def __init__(self, username='', password='', links=None):
        self.read_credentials_from_yaml()
        self.download_urls = links if links else []

    @property
    def download_urls(self):
        return self.__download_urls

    @download_urls.setter
    def download_urls(self, links):
        if links is None:
            self.__download_urls = []
        else:
            for item in links:
                try:
                    self.get_filename(item)
                except AttributeError:
                    raise ValueError(
                        'The URL seems to not have the right structure: ', item)
            self.__download_urls = links

    def set_username_and_password(self, username, password):
        self.__username = username
        self.__password = password

    def read_credentials_from_yaml(self):
        with open('Credentials.yml', 'r') as f:
            credentials = yaml.safe_load(f)
            Credentials = credentials['Credentials']
            self.set_username_and_password(
                Credentials['username'], Credentials['password'])

    def _mp_download_wrapper(self, url_item):
        """
        Wrapper for parallel download. Downloads a file from a URL and saves it.
        :param url_item: URL to download
        :type url_item: str
        """
        file_path = os.path.join(
            self.download_path, self.get_filename(url_item))
        self.__download_and_save_file(url_item, file_path)

    def start_download(self, nr_of_threads=4):
        """
        Start downloading files using multiple threads.
        :param nr_of_threads: Number of threads to use
        :type nr_of_threads: int
        """
        if self._authenticated_session is None:
            self._authenticated_session = self.__create_authenticated_session()
        os.makedirs(self.download_path, exist_ok=True)
        p = Threadpool(nr_of_threads)
        p.map(self._mp_download_wrapper, self.download_urls)
        p.close()
        p.join()

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
        s.headers = {'User-Agent': 'SuSSE'}
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
            authentication_url = self.download_urls[0]
            result = opener.open(authentication_url)
        except error.HTTPError:
            raise ValueError('Username and/or Password are incorrect!')
        except IndexError:
            raise IndexError('download_urls is not set')

        return auth_cookie_jar

    def __download_and_save_file(self, url, file_path):
        """
        Downloads and saves the file to a local directory.
        :param url: The download URL
        :param file_path: Path to save the file
        :type url: str
        :type file_path: str
        """
        response = self._authenticated_session.get(url, stream=True)
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    def __download_and_load_file(self, url):
        """
        Downloads and loads the file into memory.
        :param url: The download URL
        :return: Data as a numpy array
        """
        response = self._authenticated_session.get(url, stream=True)
        data = BytesIO(response.content)
        dataset = nc.Dataset('dummy', memory=data.read())
        array_data = np.array(dataset.variables['product_name'][:])
        return array_data
