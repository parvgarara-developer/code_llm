from pathlib import Path
import json
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

text_file = PROCESSED_DIR / "combined_text.txt"
vocab_file = PROCESSED_DIR / "char_vocab.json"
tokens_file = PROCESSED_DIR / "char_tokens.pt"

text = text_file.read_text(encoding="utf-8")

chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}

vocab_file.write_text(
    json.dumps({"stoi": stoi}, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

tokens = torch.tensor([stoi[c] for c in text], dtype=torch.long)
torch.save(tokens, tokens_file)

print("Vocab size:", len(chars))
print("Saved vocab:", vocab_file)
print("Saved tokens:", tokens_file)
print("Token count:", len(tokens))