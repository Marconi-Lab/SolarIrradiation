import os
import pandas as pd


def check_existing_coordinates(csv_filename, precision=4):
    """
    Check for existing coordinates in a CSV file to avoid duplicate downloads.
    
    Args:
        csv_filename (str): Path to the CSV file
        precision (int): Number of decimal places for coordinate comparison (default: 4)
        
    Returns:
        tuple: (existing_coordinates_set, first_write_flag)
            - existing_coordinates_set: Set of (lat, lon) tuples already downloaded
            - first_write_flag: Boolean indicating if this is the first write to the file
    """
    existing_coordinates = set()
    first_write = True
    
    if os.path.exists(csv_filename):
        try:
            existing_df = pd.read_csv(csv_filename)
            if not existing_df.empty and 'latitude' in existing_df.columns and 'longitude' in existing_df.columns:
                # Convert to numeric in case they're strings and get unique coordinate pairs
                existing_df['latitude'] = pd.to_numeric(existing_df['latitude'], errors='coerce')
                existing_df['longitude'] = pd.to_numeric(existing_df['longitude'], errors='coerce')
                
                # Remove rows where conversion failed
                existing_df = existing_df.dropna(subset=['latitude', 'longitude'])
                
                # Get unique coordinate pairs (rounded to specified precision for comparison)
                existing_coords = existing_df[['latitude', 'longitude']].drop_duplicates()
                existing_coordinates = set(
                    (round(row['latitude'], precision), round(row['longitude'], precision)) 
                    for _, row in existing_coords.iterrows()
                )
                first_write = False
                
                print(f"Found existing CSV file with {len(existing_coordinates)} unique coordinate pairs already downloaded.")
                print("Existing coordinates:")
                for coord in sorted(existing_coordinates):
                    print(f"  ({coord[0]:.{precision}f}, {coord[1]:.{precision}f})")
            else:
                print("Found existing CSV file but it's empty or missing coordinate columns.")
        except Exception as e:
            print(f"Warning: Could not read existing CSV file: {e}")
            print("Will start fresh download.")
    else:
        print("No existing CSV file found. Starting fresh download.")
    
    return existing_coordinates, first_write


def should_skip_coordinate(lat, lon, existing_coordinates, precision=4):
    """
    Check if a coordinate pair should be skipped (already exists).
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        existing_coordinates (set): Set of existing coordinate pairs
        precision (int): Number of decimal places for coordinate comparison (default: 4)
        
    Returns:
        bool: True if coordinate should be skipped, False otherwise
    """
    coord_pair = (round(lat, precision), round(lon, precision))
    return coord_pair in existing_coordinates


def add_coordinate_to_existing(lat, lon, existing_coordinates, precision=4):
    """
    Add a coordinate pair to the existing coordinates set.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        existing_coordinates (set): Set of existing coordinate pairs
        precision (int): Number of decimal places for coordinate comparison (default: 4)
    """
    coord_pair = (round(lat, precision), round(lon, precision))
    existing_coordinates.add(coord_pair)


def print_download_summary(skipped_count, processed_count, csv_filename):
    """
    Print a summary of the download process.
    
    Args:
        skipped_count (int): Number of coordinates skipped
        processed_count (int): Number of coordinates processed
        csv_filename (str): Path to the CSV file
    """
    print(f"\n{'-'*40}\nDownload complete.")
    print(f"Skipped {skipped_count} already downloaded coordinates.")
    print(f"Processed {processed_count} new coordinates.")
    print(f"Data has been written to {csv_filename}")


def show_data_preview(csv_filename, first_write):
    """
    Show a preview of the data in the CSV file.
    
    Args:
        csv_filename (str): Path to the CSV file
        first_write (bool): Whether this was the first write to the file
    """
    if first_write == False:  # Only if we wrote some data
        print("\n--- Data Preview ---")
        try:
            preview_df = pd.read_csv(csv_filename)
            print(preview_df.head())
        except Exception as e:
            print(f"Could not read preview: {e}")
        print("--------------------")
