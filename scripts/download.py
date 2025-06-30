import numpy as np
import pandas as pd 
from datetime import datetime
import time
import requests
from geopy import Point

from susse import NASAPowerProduct, TemporalResolution, NASAPowerFetchData


# Uganda bounding box (min_lon, max_lon, min_lat, max_lat)
min_lon, max_lon = 29.5, 35.0
min_lat, max_lat = -1.5,  4.5

# Native grid resolution (in degrees)
d_lon, d_lat = 1, 1

# Compute the center of each cell
lons = np.arange(min_lon + d_lon/2, max_lon, d_lon)
lats = np.arange(min_lat + d_lat/2, max_lat, d_lat)

# Build (lon,lat) pairs
grid_points = [(float(lon), float(lat)) for lat in lats for lon in lons]
print(f"Total grid cells to download: {len(grid_points)}")

start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)
print(f"Fetching data for the period: {start_date.date()} to {end_date.date()}")

# --- REVISED logic to perform the grid download ---

# 1. Initialize the data fetcher class
fetcher = NASAPowerFetchData()

# 2. Define the list of data products to download
products_to_download = [
    NASAPowerProduct.GHI,
    NASAPowerProduct.DNI, 
    NASAPowerProduct.DHI,
]
temporal_res = TemporalResolution.DAILY

# 3. Create a list to store the DataFrames from all grid points
all_dataframes = []

print(f"Downloading products: {[p.name for p in products_to_download]}")
print("-" * 40)

# 4. Loop through each grid point and download the data
for i, point in enumerate(grid_points):
    lon, lat = point
    print(f"\nProcessing grid point {i+1}/{len(grid_points)}: (Lat={lat:.2f}, Lon={lon:.2f})")

    location = Point(latitude=lat, longitude=lon)

    try:
        # Use fetch_multiple_parameters to get all products in one call
        multi_result = fetcher.fetch_multiple_parameters(
            temporal_resolution=temporal_res,
            start_date=start_date,
            end_date=end_date,
            location=location,
            products=products_to_download,
        )

        # Convert the multi-product result directly to a DataFrame
        df = multi_result.to_dataframe()

        # Add location info to the DataFrame
        df['latitude'] = lat
        df['longitude'] = lon

        all_dataframes.append(df)
        print(f"  \u2713 Success! Converted result to DataFrame with {len(df)} rows.")

    except requests.exceptions.HTTPError as http_err:
        print(f"  \u2717 FAILED. HTTP Error: {http_err}")
    except Exception as e:
        print(f"  \u2717 FAILED. An unexpected error occurred: {e}")

    time.sleep(1) # Pause to be respectful to the API

print(f"\n{'-'*40}\nDownload complete.")
print(f"Successfully processed {len(all_dataframes)} out of {len(grid_points)} grid points.")


# --- NEW pandas-based method to save all results to a single CSV file ---
csv_filename = "NASA_power_grid_data.csv"
print(f"\nConsolidating all data and saving to {csv_filename}...")

if not all_dataframes:
    print("No data was downloaded, so no CSV file will be created.")
else:
    # Concatenate all individual DataFrames into one large DataFrame
    final_df = pd.concat(all_dataframes)

    # Reorder columns for clarity (optional, but good practice)
    cols = ['latitude', 'longitude'] + [p.name for p in products_to_download]
    final_df = final_df.reindex(columns=cols)

    # Save the final DataFrame to a CSV file
    # The DataFrame's index (timestamp) will be written as the first column by default
    final_df.to_csv(csv_filename)
    print(f"\n\u2713 Successfully saved all data to {csv_filename}")
    print("\n--- Data Preview ---")
    print(final_df.head())
    print("--------------------")
