from pathlib import Path
import json
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SHARD_DIR = PROCESSED_DIR / "shards"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SHARD_DIR.mkdir(parents=True, exist_ok=True)

OUT_VOCAB = PROCESSED_DIR / "vocab.json"
OUT_META = PROCESSED_DIR / "dataset_meta.json"

SHARD_SIZE = 2_000_000

def safe_read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def stringify(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(str(x).strip() for x in value if str(x).strip())
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()

def looks_like_python_code(text: str) -> bool:
    text = text.strip()
    markers = ["def ", "class ", "import ", "from ", "print(", "return ", "if __name__"]
    return any(m in text for m in markers)

def extract_codesearchnet(obj):
    code = stringify(obj.get("code", ""))
    docstring = stringify(obj.get("docstring", ""))
    func_name = stringify(obj.get("func_name", ""))
    language = stringify(obj.get("language", "")).lower()

    if language and language != "python":
        return ""
    if not code:
        return ""
    if not looks_like_python_code(code):
        return ""

    parts = []
    parts.append("### Instruction:")
    if docstring:
        parts.append(docstring)
    elif func_name:
        parts.append(f"Write or explain the Python function {func_name}.")
    else:
        parts.append("Write the following Python function.")

    parts.append("\n### Response:")
    parts.append(code)

    return "\n".join(parts).strip()

def extract_opencode(obj):
    prompt_keys = [
        "instruction", "prompt", "question", "input", "problem", "query"
    ]
    answer_keys = [
        "output", "response", "answer", "solution", "code", "generated_solution"
    ]
    test_keys = ["tests", "test", "unit_tests"]

    prompt = ""
    answer = ""
    tests = ""

    for k in prompt_keys:
        val = stringify(obj.get(k, ""))
        if val:
            prompt = val
            break

    for k in answer_keys:
        val = stringify(obj.get(k, ""))
        if val:
            answer = val
            break

    for k in test_keys:
        val = stringify(obj.get(k, ""))
        if val:
            tests = val
            break

    if not prompt or not answer:
        return ""

    if not looks_like_python_code(answer):
        return ""

    parts = [
        "### Instruction:",
        prompt,
        "\n### Response:",
        answer
    ]

    if tests:
        parts.extend(["\n### Tests:", tests])

    return "\n".join(parts).strip()

def detect_sample(obj):
    if not isinstance(obj, dict):
        return ""

    keys = set(obj.keys())

    if "code" in keys and ("docstring" in keys or "code_tokens" in keys or "docstring_tokens" in keys):
        return extract_codesearchnet(obj)

    possible_instruction_keys = {
        "instruction", "prompt", "question", "input", "problem", "query",
        "output", "response", "answer", "solution", "generated_solution"
    }
    if keys & possible_instruction_keys:
        return extract_opencode(obj)

    if "messages" in keys and isinstance(obj["messages"], list):
        user_text = ""
        assistant_text = ""
        for m in obj["messages"]:
            if not isinstance(m, dict):
                continue
            role = stringify(m.get("role", "")).lower()
            content = stringify(m.get("content", ""))
            if role == "user" and content and not user_text:
                user_text = content
            elif role == "assistant" and content and not assistant_text:
                assistant_text = content
        if user_text and assistant_text and looks_like_python_code(assistant_text):
            return f"### Instruction:\n{user_text}\n\n### Response:\n{assistant_text}"

    return ""

def iter_all_samples():
    jsonl_files = sorted(RAW_DIR.rglob("*.jsonl"))
    json_files = sorted(RAW_DIR.rglob("*.json"))

    if not jsonl_files and not json_files:
        raise FileNotFoundError(f"No .json or .jsonl files found in {RAW_DIR}")

    for path in jsonl_files:
        count = 0
        for obj in iter_jsonl(path):
            sample = detect_sample(obj)
            if sample:
                count += 1
                yield str(path), sample + "\n\n<|END_OF_SAMPLE|>\n\n"
        print(f"Processed {path} -> {count} usable samples")

    for path in json_files:
        count = 0
        data = safe_read_json(path)

        items = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            if isinstance(data.get("data"), list):
                items = data["data"]
            else:
                items = [data]

        for obj in items:
            sample = detect_sample(obj)
            if sample:
                count += 1
                yield str(path), sample + "\n\n<|END_OF_SAMPLE|>\n\n"
        print(f"Processed {path} -> {count} usable samples")

# pass 1: build vocab and counts
vocab_chars = set()
file_counts = {}
sample_count = 0

for file_path, sample in iter_all_samples():
    vocab_chars.update(sample)
    file_counts[file_path] = file_counts.get(file_path, 0) + 1
    sample_count += 1

if sample_count == 0:
    raise ValueError("No training texts extracted. Check your dataset keys/format.")

chars = sorted(vocab_chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

OUT_VOCAB.write_text(
    json.dumps(
        {
            "stoi": stoi,
            "itos": {str(i): ch for i, ch in itos.items()},
            "vocab_size": len(chars),
            "num_samples": sample_count,
        },
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)

OUT_META.write_text(
    json.dumps(file_counts, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

for old in SHARD_DIR.glob("tokens_*.pt"):
    old.unlink()

buffer = []
shard_idx = 0

def flush_shard():
    global shard_idx
    if not buffer:
        return
    out_file = SHARD_DIR / f"tokens_{shard_idx:05d}.pt"
    torch.save(torch.tensor(buffer, dtype=torch.long), out_file)
    print(f"Saved {out_file} -> {len(buffer)} tokens")
    buffer.clear()
    shard_idx += 1

# pass 2: encode and shard
for _, sample in iter_all_samples():
    for ch in sample:
        buffer.append(stoi[ch])
        if len(buffer) >= SHARD_SIZE:
            flush_shard()

flush_shard()

print("\nDone.")
print("Total usable samples:", sample_count)
print("Vocab size:", len(chars))
print("Total shards:", shard_idx)
print("Saved vocab:", OUT_VOCAB)
print("Saved metadata:", OUT_META)