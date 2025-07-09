import numpy as np
import pandas as pd 
from datetime import datetime
import time
import requests
import os
from geopy import Point as GeoPoint
import geopandas as gpd
from shapely.geometry import Point

from susse import NASAPowerProduct, TemporalResolution, NASAPowerFetchData
from download_utils import (
    check_existing_coordinates, 
    should_skip_coordinate, 
    add_coordinate_to_existing,
    print_download_summary,
    show_data_preview
)

GRID_SPACING_DEGREES = 0.1

UGANDA_SHAPEFILE = "jupyter_notebooks/Data/gadm41_UGA_shp/gadm41_UGA_0.shp"
uganda_map = gpd.read_file(UGANDA_SHAPEFILE)
uganda_map = uganda_map.to_crs(epsg=4326)
uganda_boundary = uganda_map.union_all() 
uganda_boundary = uganda_boundary.buffer(0) 

minx, miny, maxx, maxy = uganda_boundary.bounds

# Create a grid of points within the Uganda boundary
lon_grid = np.arange(minx, maxx, GRID_SPACING_DEGREES)
lat_grid = np.arange(miny, maxy, GRID_SPACING_DEGREES)
lons, lats = np.meshgrid(lon_grid, lat_grid)
grid_points = [(lon, lat) for lon, lat in zip(lons.flatten(), lats.flatten())]

# Filter grid points to only include those within the Uganda boundary
grid_point_objects = [Point(lon, lat) for lon, lat in grid_points]
grid_gdf = gpd.GeoDataFrame(geometry=grid_point_objects, crs=uganda_map.crs)
grid_gdf = grid_gdf[grid_gdf.within(uganda_boundary)]

# Extract coordinates array for points within Uganda boundary
uganda_coordinates = [(point.geometry.x, point.geometry.y) for _, point in grid_gdf.iterrows()]
print(f"Number of coordinates within Uganda boundary: {len(uganda_coordinates)}")

start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)
print(f"Fetching data for the period: {start_date.date()} to {end_date.date()}")

# Initialize the data fetcher class
fetcher = NASAPowerFetchData()

# Define the list of data products to download
products_to_download = [
    NASAPowerProduct.GHI,
    NASAPowerProduct.DNI, 
    NASAPowerProduct.DHI,
]
temporal_res = TemporalResolution.DAILY

# Create CSV file for direct writing and check existing data
csv_filename = "scripts/NASA_power_ug_data.csv"
existing_coordinates, first_write = check_existing_coordinates(csv_filename)

print(f"Downloading products: {[p.name for p in products_to_download]}")
print("-" * 40)

# Loop through each grid point and download the data
skipped_count = 0
processed_count = 0

for i, point in enumerate(uganda_coordinates[:3]):  # Limiting to first 3 points for demonstration
    lon, lat = point
    
    # Check if this coordinate pair already exists
    if should_skip_coordinate(lat, lon, existing_coordinates):
        skipped_count += 1
        print(f"\nSkipping grid point {i+1}/{len(uganda_coordinates)}: (Lat={lat:.2f}, Lon={lon:.2f}) - Already downloaded")
        continue
    
    processed_count += 1
    print(f"\nProcessing grid point {i+1}/{len(uganda_coordinates)}: (Lat={lat:.2f}, Lon={lon:.2f})")

    location = GeoPoint(latitude=lat, longitude=lon)

    try:
        multi_result = fetcher.fetch_multiple_parameters(
            temporal_resolution=temporal_res,
            start_date=start_date,
            end_date=end_date,
            location=location,
            products=products_to_download,
        )

        df = multi_result.to_dataframe()

        df['latitude'] = lat
        df['longitude'] = lon
        
        cols = ['latitude', 'longitude'] + [p.name for p in products_to_download]
        df = df.reindex(columns=cols)
        
        df.to_csv(csv_filename, mode='a', header=first_write, index=True)
        first_write = False

        add_coordinate_to_existing(lat, lon, existing_coordinates)

        print(f"  \u2713 Success! Converted result to DataFrame with {len(df)} rows and wrote to {csv_filename}.")

    except requests.exceptions.HTTPError as http_err:
        print(f"  \u2717 FAILED. HTTP Error: {http_err}")
    except Exception as e:
        print(f"  \u2717 FAILED. An unexpected error occurred: {e}")

    time.sleep(60)

print_download_summary(skipped_count, processed_count, csv_filename)
show_data_preview(csv_filename, first_write)
