import os
import requests
import zipfile
from pathlib import Path

def download_openmoji():
    """
    Download OpenMoji dataset and extract to ./emojis directory.
    """
    # OpenMoji PNG files URL (from GitHub releases)
    url = "https://github.com/hfg-gmuend/openmoji/releases/latest/download/openmoji-72x72-color.zip"
    zip_path = "./openmoji.zip"
    extract_path = "./emojis"
    
    # Create extract directory if it doesn't exist
    Path(extract_path).mkdir(parents=True, exist_ok=True)
    
    print("Downloading OpenMoji dataset...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Save the zip file
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print("Extracting OpenMoji dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    # Clean up
    os.remove(zip_path)
    
    # Count the number of PNG files
    png_files = list(Path(extract_path).glob("**/*.png"))
    print(f"Downloaded {len(png_files)} OpenMoji PNG files to {extract_path}")

if __name__ == "__main__":
    download_openmoji()
