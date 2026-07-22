from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "python_dataset" / "python" / "python" / "final" / "jsonl" / "train"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = PROCESSED_DIR / "codesearchnet_text.txt"
END_TOKEN = "<|end|>"

jsonl_files = list(RAW_DIR.glob("*.jsonl"))
if not jsonl_files:
    raise FileNotFoundError(f"No .jsonl files found in {RAW_DIR}")

samples = []
kept = 0
skipped = 0

for file in jsonl_files:
    print("Reading:", file)
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            doc = str(row.get("docstring") or "").strip()
            code = str(row.get("code") or "").strip()

            if not doc or not code:
                skipped += 1
                continue

            if len(code) < 40:
                skipped += 1
                continue

            sample = (
                f"### Instruction:\n{doc}\n\n"
                f"### Response:\n{code}\n"
                f"{END_TOKEN}"
            )
            samples.append(sample)
            kept += 1

OUT_FILE.write_text("\n\n".join(samples), encoding="utf-8")

print("Kept:", kept)
print("Skipped:", skipped)
print("Saved:", OUT_FILE)
print("Size bytes:", OUT_FILE.stat().st_size if OUT_FILE.exists() else 0)