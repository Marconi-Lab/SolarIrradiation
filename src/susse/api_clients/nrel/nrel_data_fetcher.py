import requests
import time
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, TextIO
import pandas as pd

class NRELDataFetcher:
    """A class to fetch solar irradiation data from NREL's API"""
    
    BASE_URL = "https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-msg-v1-0-0-download.json?"
    RETRY_ATTEMPTS = 7
    CHUNK_SIZE = 8192
    
    def __init__(self, credentials_path: str):
        """Initialize the NREL Data Fetcher with credentials"""
        self.credentials = self._load_credentials(credentials_path)
        self.api_key = self.credentials['nrel']['api_key']
        self.email = self.credentials['nrel']['email']

    @staticmethod
    def _load_credentials(credentials_path: str) -> dict:
        """Load credentials from YAML file"""
        with open(credentials_path, 'r') as file:
            return yaml.safe_load(file)

    def _get_default_attributes(self) -> str:
        """Return default attributes for the API request"""
        return (
            'solar_zenith_angle,surface_albedo,total_precipitable_water,'
            'clearsky_dhi,clearsky_dni,clearsky_ghi,cloud_type,dew_point,'
            'relative_humidity,surface_pressure,dhi,dni,fill_flag,ghi,'
            'air_temperature,wind_direction,wind_speed'
        )

    def _create_request_payload(self, year: int, location_id: str) -> dict:
        """Create the request payload for the API"""
        return {
            'attributes': self._get_default_attributes(),
            'interval': '60',
            'api_key': self.api_key,
            'email': self.email,
            'names': [year],
            'location_ids': location_id
        }

    def _get_response_json(self, response: requests.Response) -> dict:
        """Process API response and handle errors"""
        if response.status_code != 200:
            raise requests.exceptions.HTTPError(
                f"Server error: {response.status_code} {response.reason}\n"
                f"Response: {response.text}"
            )

        try:
            response_json = response.json()
        except ValueError:
            raise ValueError(f"Invalid JSON response: {response.text}")

        if response_json.get('errors'):
            raise ValueError(f"API errors: {'\n'.join(response_json['errors'])}")
            
        return response_json

    def _download_file(self, url: str, local_filename: str) -> TextIO:
        """Download file from URL with retry mechanism"""
        for attempt in range(self.RETRY_ATTEMPTS):
            try:
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(local_filename, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=self.CHUNK_SIZE):
                            f.write(chunk)
                    return f
            except requests.exceptions.HTTPError as e:
                if attempt == self.RETRY_ATTEMPTS - 1:
                    raise e
                print(f'Download attempt {attempt + 1} failed, retrying...')
                time.sleep(2 ** attempt)

    def fetch_data(self, years: List[int], locations: List[str]) -> Dict[Tuple[int, str], TextIO]:
        """
        Fetch solar data for specified years and locations
        
        Parameters
        ----------
        years : List[int]
            List of years to fetch data for
        locations : List[str]
            List of location IDs
            
        Returns
        -------
        Dict[Tuple[int, str], TextIO]
            Dictionary mapping (year, location) to downloaded file handlers
        """
        files = {}
        headers = {'x-api-key': self.api_key}

        for year in years:
            print(f"Processing year: {year}")
            for idx, location_id in enumerate(locations, 1):
                print(f'Processing location {idx} of {len(locations)}...')
                
                payload = self._create_request_payload(year, location_id)
                response = requests.post(self.BASE_URL, payload, headers=headers)
                data = self._get_response_json(response)
                
                download_url = data['outputs']['downloadUrl']
                print(data['outputs']['message'])
                print(f"Download URL: {download_url}")

                filename = f"{year}_{location_id}.zip"
                files[(year, location_id)] = self._download_file(download_url, filename)
                
                time.sleep(1)  # Rate limiting
                print(f'Processed location {location_id}')

        return files