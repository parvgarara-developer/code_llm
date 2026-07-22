from pathlib import Path
import json
import torch
from model_gpt import GPTModel

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR = BASE_DIR / "output"

CHECKPOINT_PATH = CHECKPOINT_DIR / "gpt_opencode.pt"
CONFIG_PATH = CHECKPOINT_DIR / "gpt_opencode_config.json"
VOCAB_PATH = PROCESSED_DIR / "char_vocab.json"
OUTPUT_PATH = OUTPUT_DIR / "evaluation_outputs.txt"

device = "cpu"
END_TOKEN = "<|end|>"

prompts = [
    "Write a Python function to add two numbers.",
    "Write a Python function to check if a string is a palindrome.",
    "Write a Python function to return the factorial of a number.",
    "Write a Python function to find the maximum element in a list.",
    "Write a Python function to count vowels in a string.",
    "Write a Python function to reverse a list without using built-in reverse().",
    "Write a Python function to remove duplicates from a list while keeping order.",
    "Write a Python function to check whether a number is prime.",
    "Write a Python function to merge two dictionaries.",
    "Write a Python function to compute the sum of digits of an integer."
]

if not VOCAB_PATH.exists():
    raise FileNotFoundError(f"Missing vocab file: {VOCAB_PATH}")
if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Missing checkpoint file: {CHECKPOINT_PATH}")
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
stoi = vocab["stoi"]
itos = {i: ch for ch, i in stoi.items()}

model_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
model = GPTModel(**model_config).to(device)

state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

results = []

for i, task in enumerate(prompts, start=1):
    prompt = f"### Instruction:\n{task}\n\n### Response:\n"
    filtered_prompt = "".join(ch for ch in prompt if ch in stoi)

    if not filtered_prompt:
        results.append(f"Prompt {i}: {task}\nERROR: Prompt contains no known vocabulary characters.\n")
        continue

    idx = torch.tensor([[stoi[ch] for ch in filtered_prompt]], dtype=torch.long, device=device)

    with torch.no_grad():
        output_ids = model.generate(
            idx,
            max_new_tokens=220,
            temperature=0.2,
            top_k=10
        )

    tokens = output_ids[0].tolist()
    text = "".join(itos.get(int(t), "") for t in tokens)

    if END_TOKEN in text:
        text = text.split(END_TOKEN)[0]

    block = (
        f"{'=' * 80}\n"
        f"Prompt {i}\n"
        f"Task: {task}\n"
        f"{'-' * 80}\n"
        f"{text.strip()}\n\n"
    )
    results.append(block)

OUTPUT_PATH.write_text("".join(results), encoding="utf-8")

print(f"Saved evaluation outputs to: {OUTPUT_PATH}")