from pathlib import Path
import zipfile

zip_path = Path("data/raw/python.zip")
extract_dir = Path("data/raw/python_dataset")

if not zip_path.exists():
    raise FileNotFoundError(f"ZIP not found: {zip_path}")

extract_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extract_dir)

print(f"Extracted to: {extract_dir.resolve()}")

gz_files = list(extract_dir.rglob("*.jsonl.gz"))
print(f"Found {len(gz_files)} .jsonl.gz files after extraction")

for f in gz_files[:20]:
    print(f)