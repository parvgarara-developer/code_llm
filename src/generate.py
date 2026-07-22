from pathlib import Path
import json
import torch
from model_gpt import GPTModel

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

CHECKPOINT_PATH = CHECKPOINT_DIR / "gpt_opencode.pt"
CONFIG_PATH = CHECKPOINT_DIR / "gpt_opencode_config.json"
VOCAB_PATH = PROCESSED_DIR / "char_vocab.json"

device = "cpu"
END_TOKEN = "<|end|>"

if not VOCAB_PATH.exists():
    raise FileNotFoundError(f"Missing vocab file: {VOCAB_PATH}")
if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Missing checkpoint file: {CHECKPOINT_PATH}")
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")

vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
stoi = vocab["stoi"]
itos = {i: ch for ch, i in stoi.items()}

model_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
model = GPTModel(**model_config).to(device)

state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

prompt = "### Instruction:\nWrite a Python function to count vowels in a string.\n\n### Response:\n"
filtered_prompt = "".join(ch for ch in prompt if ch in stoi)

if not filtered_prompt:
    raise ValueError("Prompt contains no characters from the training vocabulary.")

idx = torch.tensor([[stoi[ch] for ch in filtered_prompt]], dtype=torch.long, device=device)

with torch.no_grad():
    output_ids = model.generate(
        idx,
        max_new_tokens=220,
        temperature=0.2,
        top_k=10
    )

tokens = output_ids[0].tolist()
text = "".join(itos.get(int(i), "") for i in tokens)

if END_TOKEN in text:
    text = text.split(END_TOKEN)[0]

print(text)