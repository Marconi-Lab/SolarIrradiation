import numpy as np
import pandas as pd
from datetime import datetime, timezone
import time
import os
import geopandas as gpd
from shapely.geometry import Point

from susse import CAMSClient
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

start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
end_date = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

time_step = '1d'
print(f"Fetching data for the period: {start_date.date()} (UTC)")
print(f"Time step: {time_step}")

client = CAMSClient()

print("-" * 40)

csv_filename = "scripts/cams_radiation_ug_data.csv"
existing_coordinates, first_write = check_existing_coordinates(csv_filename)

# Loop through each grid point and download the data
skipped_count = 0
processed_count = 0
for i, point in enumerate(uganda_coordinates): 
    lon, lat = point
    
    if should_skip_coordinate(lat, lon, existing_coordinates):
        skipped_count += 1
        print(f"\nSkipping grid point {i+1}/{len(uganda_coordinates)}: (Lat={lat:.2f}, Lon={lon:.2f}) - Already downloaded")
        continue
    
    processed_count += 1
    print(f"\nProcessing grid point {i+1}/{len(uganda_coordinates)}: (Lat={lat:.2f}, Lon={lon:.2f})")

    result = client.fetch_data(
        latitude=lat,
        longitude=lon,
        start=start_date,
        end=end_date,
        time_step=time_step,
    )

    if result["error"]:
        print(f"  \u2717 FAILED. An error occurred: {result['error']}")
    else:
        df = pd.DataFrame(result["data"])
        
        df['latitude'] = lat
        df['longitude'] = lon
        
        data_cols = [col for col in df.columns if col not in ['latitude', 'longitude']]
        df = df[['latitude', 'longitude'] + data_cols]
        
        df.to_csv(csv_filename, mode='a', header=first_write, index=False)
        first_write = False
        
        add_coordinate_to_existing(lat, lon, existing_coordinates)
        
        print(f"  \u2713 Success! Fetched and wrote {len(df)} records to {csv_filename}.")

    time.sleep(900)  # Make 4 requests per hour since CAMS API has a limit of 100 requests per day

print_download_summary(skipped_count, processed_count, csv_filename)
show_data_preview(csv_filename, first_write)
