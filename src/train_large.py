from pathlib import Path
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SHARD_DIR = PROCESSED_DIR / "shards"
VOCAB_FILE = PROCESSED_DIR / "vocab.json"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

if not VOCAB_FILE.exists():
    raise FileNotFoundError(f"Missing vocab file: {VOCAB_FILE}")
if not SHARD_DIR.exists():
    raise FileNotFoundError(f"Missing shard directory: {SHARD_DIR}")

vocab = json.loads(VOCAB_FILE.read_text(encoding="utf-8"))
stoi = vocab["stoi"]
vocab_size = len(stoi)

shard_files = sorted(SHARD_DIR.glob("tokens_*.pt"))
if not shard_files:
    raise FileNotFoundError(f"No shard files found in: {SHARD_DIR}")

block_size = 64
batch_size = 64
max_iters = 10000
eval_interval = 100
eval_iters = 25
learning_rate = 10e-4
grad_clip = 1.0
dropout = 0.1
n_embd = 512
n_head = 8
n_layer = 8

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLarge(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.reshape(B * T, C)
            targets = targets.reshape(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = GPTLarge().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

def load_shard(path):
    data = torch.load(path, map_location="cpu")
    if not torch.is_tensor(data):
        data = torch.tensor(data, dtype=torch.long)
    data = data.long()
    if data.numel() <= block_size + 1:
        return None
    return data

def get_batch(data):
    ix = torch.randint(0, data.numel() - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    sample_shards = shard_files[:min(3, len(shard_files))]
    for split in ["train", "val"]:
        losses = []
        chosen = sample_shards if split == "train" else shard_files[-min(2, len(shard_files)):]
        for shard_path in chosen:
            data = load_shard(shard_path)
            if data is None:
                continue
            for _ in range(eval_iters):
                xb, yb = get_batch(data)
                _, loss = model(xb, yb)
                losses.append(loss.item())
        out[split] = sum(losses) / max(len(losses), 1)
    model.train()
    return out

for step in range(max_iters):
    shard_path = random.choice(shard_files)
    data = load_shard(shard_path)
    if data is None:
        continue

    xb, yb = get_batch(data)
    _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step}: train {losses['train']:.4f}, val {losses['val']:.4f}")

checkpoint_path = CHECKPOINT_DIR / "gpt_large_sharded.pt"
torch.save(model.state_dict(), checkpoint_path)
print("Saved checkpoint to:", checkpoint_path)
