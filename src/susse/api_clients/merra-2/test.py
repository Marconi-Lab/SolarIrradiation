from merra_config import Merra2Config


# Example parameters
download_years = ['01-10-2020', '01-12-2021']
base_url = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2T1NXSLV.5.12.4/'
dataset_name = 'tavg1_2d_slv_Nx'
url_params = 'T2M[0:1:23][0:360][0:575]'

# Generate download links
links = Merra2Config.generate_download_links(
    download_years, base_url, dataset_name, url_params)
print(links)


# Generate URL parameters
params = Merra2Config.generate_url_params(
    ['T2M'], '[0:1:23]', '[0:360]', '[0:575]')
print(params)
