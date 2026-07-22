from pathlib import Path
import requests
import zipfile

URL = "https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip"
RAW_DIR = Path("data/raw")
ZIP_PATH = RAW_DIR / "python.zip"
EXTRACT_DIR = RAW_DIR / "python_dataset"

RAW_DIR.mkdir(parents=True, exist_ok=True)

print("RAW_DIR:", RAW_DIR.resolve())
print("ZIP_PATH:", ZIP_PATH.resolve())
print("EXTRACT_DIR:", EXTRACT_DIR.resolve())

if not ZIP_PATH.exists():
    print("Downloading CodeSearchNet Python...")
    with requests.get(URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(ZIP_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print("Download complete:", ZIP_PATH)
else:
    print("ZIP already exists:", ZIP_PATH)

if EXTRACT_DIR.exists():
    print("Extraction folder already exists, skipping extraction.")
else:
    print("Extracting zip...")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)
    print("Extraction complete.")

all_files = list(EXTRACT_DIR.rglob("*"))
print(f"Extracted total entries: {len(all_files)}")

gz_files = list(EXTRACT_DIR.rglob("*.jsonl.gz"))
print(f"Found gz files after extraction: {len(gz_files)}")

for p in gz_files[:10]:
    print("GZ:", p)