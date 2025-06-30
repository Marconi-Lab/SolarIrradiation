import numpy as np
import pandas as pd
from datetime import datetime, timezone
import time

from susse import CAMSClient

# --- Grid Definition (Same as before) ---

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
print(f"Total grid cells to process: {len(grid_points)}")

# --- CAMS Download Configuration ---

# CAMS data is in UTC. It's best to use timezone-aware datetimes.
# For this example, we will download data for a single day.
start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
end_date = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

# Time step must be an ISO 8601 duration string. 'PT1H' means 1-hour intervals.
time_step = '1d'
print(f"Fetching data for the period: {start_date.date()} (UTC)")
print(f"Time step: {time_step} (1 hour)")


# --- Main Logic to Perform Grid Download ---

# 1. Initialize the CAMSClient
# This will handle getting your CAMS-registered email.
client = CAMSClient()

# 2. Create a list to store the DataFrames from all grid points
all_dataframes = []

print("-" * 40)

# 3. Loop through each grid point and download the data
for i, point in enumerate(grid_points):
    lon, lat = point
    print(f"\nProcessing grid point {i+1}/{len(grid_points)}: (Lat={lat:.2f}, Lon={lon:.2f})")

    # Fetch the data using the client's method
    result = client.fetch_data(
        latitude=lat,
        longitude=lon,
        start=start_date,
        end=end_date,
        time_step=time_step,
    )

    # The client returns a dictionary with an 'error' key. Check it first.
    if result["error"]:
        print(f"  \u2717 FAILED. An error occurred: {result['error']}")
    else:
        # The data is in a list of records under the 'data' key
        df = pd.DataFrame(result["data"])
        
        # Add location info to the DataFrame
        df['latitude'] = lat
        df['longitude'] = lon
        
        all_dataframes.append(df)
        print(f"  \u2713 Success! Fetched {len(df)} hourly records.")

    # It's good practice to add a small delay between API requests
    time.sleep(1)

print(f"\n{'-'*40}\nDownload complete.")
print(f"Successfully processed {len(all_dataframes)} out of {len(grid_points)} grid points.")


# --- Consolidate and Save to CSV ---
csv_filename = "cams_radiation_grid_data.csv"
print(f"\nConsolidating all data and saving to {csv_filename}...")

if not all_dataframes:
    print("No data was downloaded, so no CSV file will be created.")
else:
    # Concatenate all individual DataFrames into one large DataFrame
    final_df = pd.concat(all_dataframes, ignore_index=True)

    # Reorder columns for clarity (optional, but good practice)
    # Move location columns to the front
    data_cols = [col for col in final_df.columns if col not in ['latitude', 'longitude']]
    final_df = final_df[['latitude', 'longitude'] + data_cols]

    # Save the final DataFrame to a CSV file
    final_df.to_csv(csv_filename, index=False)
    
    print(f"\n\u2713 Successfully saved all data to {csv_filename}")
    print("\n--- Data Preview ---")
    print(final_df.head())
    print("--------------------")
