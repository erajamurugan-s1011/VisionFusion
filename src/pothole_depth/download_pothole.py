import os
import zipfile
import requests

DOWNLOAD_URL = "https://public.roboflow.com/ds/yXHrvtLM9V?key=6AnV2zl07L"
DEST_DIR = "../../data/pothole_full"
ZIP_PATH = "../../data/pothole_full.zip"

os.makedirs(os.path.dirname(ZIP_PATH), exist_ok=True)

print("Downloading...")
response = requests.get(DOWNLOAD_URL, stream=True)
response.raise_for_status()

with open(ZIP_PATH, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

print("Extracting...")
os.makedirs(DEST_DIR, exist_ok=True)
with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
    zip_ref.extractall(DEST_DIR)

os.remove(ZIP_PATH)
print("Downloaded and extracted to:", DEST_DIR)