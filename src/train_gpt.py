from pathlib import Path
import json
import torch
from model_gpt import GPTModel

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

device = "cpu"
print("Using device:", device)
print("Base dir:", BASE_DIR)

tokens_file = PROCESSED_DIR / "char_tokens.pt"
vocab_file = PROCESSED_DIR / "char_vocab.json"
config_file = CHECKPOINT_DIR / "gpt_opencode_config.json"
checkpoint_file = CHECKPOINT_DIR / "gpt_opencode.pt"

if not tokens_file.exists():
    raise FileNotFoundError(f"Missing token file: {tokens_file}")
if not vocab_file.exists():
    raise FileNotFoundError(f"Missing vocab file: {vocab_file}")

tokens = torch.load(tokens_file, map_location="cpu").long()
vocab = json.loads(vocab_file.read_text(encoding="utf-8"))
stoi = vocab["stoi"]
vocab_size = len(stoi)

print("Total tokens:", len(tokens))
print("Vocab size:", vocab_size)

if len(tokens) < 1000:
    raise ValueError("Token file is too small. Check preprocessing and tokenization.")

n = int(0.9 * len(tokens))
train_data = tokens[:n]
val_data = tokens[n:]

batch_size = 4
block_size = 128
max_iters = 5000
eval_interval = 50
eval_iters = 20
learning_rate = 10e-4

def get_batch(split):
    source = train_data if split == "train" else val_data
    if len(source) <= block_size + 1:
        raise ValueError(f"Not enough tokens in {split} split for block_size={block_size}")
    ix = torch.randint(0, len(source) - block_size - 1, (batch_size,))
    x = torch.stack([source[i:i + block_size] for i in ix])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

model_config = {
    "vocab_size": vocab_size,
    "block_size": 128,
    "n_embd": 128,
    "n_head": 4,
    "n_layer": 3,
    "dropout": 0.1
}

model = GPTModel(**model_config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {step}: train {losses['train']:.4f}, val {losses['val']:.4f}")

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

torch.save(model.state_dict(), checkpoint_file)
config_file.write_text(json.dumps(model_config, indent=2), encoding="utf-8")

print("Saved checkpoint:", checkpoint_file)
print("Saved config:", config_file)