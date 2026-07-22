from pathlib import Path
import json
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SHARD_DIR = PROCESSED_DIR / "shards"

SHARD_SIZE = 2_000_000

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SHARD_DIR.mkdir(parents=True, exist_ok=True)

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

def build_from_opencodeinstruct(obj):
    prompt_keys = ["instruction", "prompt", "question", "input", "problem", "query"]
    answer_keys = ["output", "response", "answer", "solution", "code", "generated_solution"]
    test_keys = ["tests", "test", "unit_tests"]

    prompt = ""
    answer = ""
    tests = ""

    for k in prompt_keys:
        if k in obj and stringify(obj[k]):
            prompt = stringify(obj[k])
            break
    for k in answer_keys:
        if k in obj and stringify(obj[k]):
            answer = stringify(obj[k])
            break
    for k in test_keys:
        if k in obj and stringify(obj[k]):
            tests = stringify(obj[k])
            break

    parts = []
    if prompt:
        parts.append("### Instruction:\n" + prompt)
    if answer:
        parts.append("### Response:\n" + answer)
    if tests:
        parts.append("### Tests:\n" + tests)
    return "\n\n".join(parts).strip()

def build_from_codesearchnet(obj):
    code = stringify(obj.get("code", ""))
    docstring = stringify(obj.get("docstring", ""))
    func_name = stringify(obj.get("func_name", ""))
    language = stringify(obj.get("language", ""))

    if not code:
        return ""

    parts = []
    if language:
        parts.append(f"### Language:\n{language}")
    if func_name:
        parts.append(f"### Function:\n{func_name}")
    if docstring:
        parts.append(f"### Description:\n{docstring}")
    parts.append(f"### Code:\n{code}")
    return "\n\n".join(parts).strip()

def detect_and_convert(obj):
    if not isinstance(obj, dict):
        return ""
    keys = set(obj.keys())

    if "code" in keys and ("docstring" in keys or "code_tokens" in keys or "docstring_tokens" in keys):
        return build_from_codesearchnet(obj)

    opencode_keys = {
        "instruction", "prompt", "question", "input",
        "output", "response", "answer", "solution", "tests", "test"
    }
    if keys & opencode_keys:
        return build_from_opencodeinstruct(obj)

    if "messages" in keys and isinstance(obj["messages"], list):
        msgs = []
        for m in obj["messages"]:
            if isinstance(m, dict):
                role = stringify(m.get("role", "user")).upper()
                content = stringify(m.get("content", ""))
                if content:
                    msgs.append(f"### {role}:\n{content}")
        return "\n\n".join(msgs).strip()

    fallback_parts = []
    for field in ["instruction", "prompt", "question", "input", "output", "response", "answer", "solution", "code", "docstring"]:
        val = stringify(obj.get(field, ""))
        if val:
            fallback_parts.append(f"### {field.capitalize()}:\n{val}")
    return "\n\n".join(fallback_parts).strip()

def iter_samples():
    jsonl_files = sorted(RAW_DIR.rglob("*.jsonl"))
    json_files = sorted(RAW_DIR.rglob("*.json"))

    if not jsonl_files and not json_files:
        raise FileNotFoundError(f"No .json or .jsonl files found in {RAW_DIR}")

    for path in jsonl_files:
        local_count = 0
        for obj in iter_jsonl(path):
            text = detect_and_convert(obj)
            if text:
                local_count += 1
                yield path, text + "\n\n<|END_OF_SAMPLE|>\n\n"
        print(f"Scanned {path} -> {local_count} samples")

    for path in json_files:
        local_count = 0
        data = safe_read_json(path)

        if isinstance(data, list):
            for obj in data:
                text = detect_and_convert(obj)
                if text:
                    local_count += 1
                    yield path, text + "\n\n<|END_OF_SAMPLE|>\n\n"
        elif isinstance(data, dict):
            if isinstance(data.get("data"), list):
                for obj in data["data"]:
                    text = detect_and_convert(obj)
                    if text:
                        local_count += 1
                        yield path, text + "\n\n<|END_OF_SAMPLE|>\n\n"
            else:
                text = detect_and_convert(data)
                if text:
                    local_count += 1
                    yield path, text + "\n\n<|END_OF_SAMPLE|>\n\n"

        print(f"Scanned {path} -> {local_count} samples")

# pass 1: build vocab
vocab_chars = set()
file_counts = {}
sample_count = 0

for path, sample in iter_samples():
    vocab_chars.update(sample)
    file_counts[str(path)] = file_counts.get(str(path), 0) + 1
    sample_count += 1

if sample_count == 0:
    raise ValueError("No usable samples were extracted from the raw datasets.")

chars = sorted(vocab_chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

vocab = {
    "stoi": stoi,
    "itos": {str(i): ch for i, ch in itos.items()},
    "vocab_size": len(chars),
    "num_samples": sample_count,
    "files": file_counts,
}
(PROCESSED_DIR / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")

# clear old shards
for old_file in SHARD_DIR.glob("tokens_*.pt"):
    old_file.unlink()

# pass 2: encode with stoi and shard
buffer = []
shard_count = 0

def flush_shard():
    global shard_count
    if not buffer:
        return
    shard_path = SHARD_DIR / f"tokens_{shard_count:05d}.pt"
    tensor = torch.tensor(buffer, dtype=torch.long)
    torch.save(tensor, shard_path)
    print(f"Saved shard {shard_path} -> {tensor.numel()} tokens")
    buffer.clear()
    shard_count += 1

for _, sample in iter_samples():
    for ch in sample:
        buffer.append(stoi[ch])
        if len(buffer) >= SHARD_SIZE:
            flush_shard()

flush_shard()

print("\nDone.")
print("Saved vocab to:", PROCESSED_DIR / "vocab.json")
print("Samples:", sample_count)
print("Chars:", len(chars))
print("Shards:", shard_count)
print("Tokens are stored in:", SHARD_DIR)