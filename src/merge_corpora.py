from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

opencode_file = PROCESSED_DIR / "opencode_text.txt"
codesearch_file = PROCESSED_DIR / "codesearchnet_text.txt"
merged_file = PROCESSED_DIR / "combined_text.txt"

texts = []

if opencode_file.exists():
    texts.append(opencode_file.read_text(encoding="utf-8"))

if codesearch_file.exists():
    texts.append(codesearch_file.read_text(encoding="utf-8"))

if not texts:
    raise FileNotFoundError("No processed corpora found to merge.")

merged_file.write_text("\n\n".join(texts), encoding="utf-8")

print("Saved merged corpus:", merged_file)
print("Size bytes:", merged_file.stat().st_size)