import os
import requests
from zipfile import ZipFile
import io
import shutil

def download_and_extract_zip():
    """Download the ZIP file from Dropbox and extract it to the data/RawData directory."""
    url = "https://www.dropbox.com/scl/fi/mwuaob48m61hvp779349r/RawData.zip?rlkey=1zhyv4z860gi5qjynrjxzbs7n&dl=1"  # Using dl=1 to force download

    # Create data directory if it doesn't exist
    raw_data_dir = os.path.join('data', 'RawData')
    os.makedirs(raw_data_dir, exist_ok=True)

    print(f"Downloading data from {url}...")

    try:
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad responses

        # Save the zip file
        zip_path = os.path.join('data', 'raw_data_download.zip')
        with open(zip_path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)

        print(f"Downloaded zip file to {zip_path}")

        # Extract the zip file
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_data_dir)

        print(f"Extracted files to {raw_data_dir}")

        # Optionally remove the zip file after extraction
        os.remove(zip_path)
        print("Download and extraction completed successfully!")

    except Exception as e:
        print(f"Error downloading or extracting the data: {e}")
        return False

    return True

if __name__ == "__main__":
    download_and_extract_zip()
