from pathlib import Path
import json
import ast

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "opencodeinstruct"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = PROCESSED_DIR / "opencode_text.txt"
END_TOKEN = "<|end|>"

def read_json_records(path):
    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            yield from obj
        elif isinstance(obj, dict):
            if "data" in obj and isinstance(obj["data"], list):
                yield from obj["data"]
            else:
                yield obj
        else:
            raise ValueError(f"Unsupported JSON format in {path}")

def get_prompt(row):
    for key in ["input", "instruction", "prompt", "question", "user_prompt"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""

def parse_structured_output(text):
    text = text.strip()
    if not text:
        return ""

    if text.startswith("{") and text.endswith("}"):
        try:
            obj = ast.literal_eval(text)
            if isinstance(obj, dict):
                for key in ["fixed_code", "code", "response", "output", "answer"]:
                    value = obj.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        except Exception:
            pass

    return text

def looks_like_python_code(text):
    text_lower = text.lower()

    python_signals = [
        "def ",
        "import ",
        "from ",
        "class ",
        "print(",
        "return ",
        "if __name__",
        "elif ",
        "except ",
        "with ",
        "lambda ",
        "for ",
        "while "
    ]

    bad_signals = [
        "select * from",
        "console.log(",
        "function ",
        "public static void main",
        "#include <",
        "package main",
        "fmt.println",
        "golang",
        "javascript",
        "typescript",
        "java ",
        "sql "
    ]

    if any(x in text_lower for x in bad_signals):
        return False

    return any(x in text_lower for x in python_signals)

files = list(RAW_DIR.glob("*.json")) + list(RAW_DIR.glob("*.jsonl"))
if not files:
    raise FileNotFoundError(f"No .json or .jsonl files found in {RAW_DIR}")

samples = []
kept = 0
skipped_empty = 0
skipped_non_python = 0

for file in files:
    print("Reading:", file)
    for row in read_json_records(file):
        prompt = get_prompt(row)
        raw_output = str(
            row.get("output")
            or row.get("response")
            or row.get("completion")
            or row.get("answer")
            or row.get("code")
            or ""
        ).strip()

        code = parse_structured_output(raw_output)

        if not prompt or not code:
            skipped_empty += 1
            continue

        if not looks_like_python_code(code):
            skipped_non_python += 1
            continue

        sample = (
            f"### Instruction:\n{prompt}\n\n"
            f"### Response:\n{code}\n"
            f"{END_TOKEN}"
        )
        samples.append(sample)
        kept += 1

OUT_FILE.write_text("\n\n".join(samples), encoding="utf-8")

print("Kept:", kept)
print("Skipped empty:", skipped_empty)
print("Skipped non-python:", skipped_non_python)
print("Saved:", OUT_FILE)
print("Size bytes:", OUT_FILE.stat().st_size if OUT_FILE.exists() else 0)