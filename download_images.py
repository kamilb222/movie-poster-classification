"""
Download movie poster images from URLs in MovieGenre.csv using multi-threading.
Saves images to ./images/ folder and outputs a cleaned CSV with only valid rows.
"""

import os
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configuration
CSV_PATH = "data/MovieGenre.csv"
OUTPUT_CSV_PATH = "data/MovieGenre_clean.csv"
IMAGES_DIR = "images"
MAX_WORKERS = 16  # Number of parallel download threads
TIMEOUT = 10  # Request timeout in seconds


def download_image(row):
    """
    Download a single image from the Poster URL.
    Returns (imdb_id, success) tuple.
    """
    imdb_id = row["imdbId"]
    poster_url = row["Poster"]
    
    # Skip if no URL
    if pd.isna(poster_url) or not poster_url.strip():
        return imdb_id, False
    
    image_path = os.path.join(IMAGES_DIR, f"{imdb_id}.jpg")
    
    # Skip if already downloaded
    if os.path.exists(image_path):
        return imdb_id, True
    
    try:
        response = requests.get(poster_url, timeout=TIMEOUT, stream=True)
        response.raise_for_status()
        
        # Check if content is actually an image
        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type:
            return imdb_id, False
        
        with open(image_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return imdb_id, True
    
    except (requests.RequestException, IOError):
        return imdb_id, False


def main():
    # Create images directory
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    # Load CSV
    print(f"Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH, encoding="latin-1")
    print(f"Total rows: {len(df)}")
    
    # Prepare rows for download
    rows = df.to_dict("records")
    
    # Track successful downloads
    successful_ids = set()
    failed_count = 0
    
    print(f"Downloading images with {MAX_WORKERS} threads...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all download tasks
        futures = {executor.submit(download_image, row): row["imdbId"] for row in rows}
        
        # Process completed downloads with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            imdb_id, success = future.result()
            if success:
                successful_ids.add(imdb_id)
            else:
                failed_count += 1
    
    print(f"\nDownload complete!")
    print(f"  Successful: {len(successful_ids)}")
    print(f"  Failed: {failed_count}")
    
    # Filter dataframe to only include successful downloads
    df_clean = df[df["imdbId"].isin(successful_ids)].copy()
    
    # Save cleaned CSV
    df_clean.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved cleaned CSV to {OUTPUT_CSV_PATH} ({len(df_clean)} rows)")


if __name__ == "__main__":
    main()
