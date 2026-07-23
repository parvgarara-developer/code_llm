"""
Generate code from the trained subword GPT.

Loads the architecture from the saved config (single source of truth via
model.py), encodes the prompt with the retrained code tokenizer, and samples
with temperature / top-k / top-p / repetition penalty, stopping at EOS.

  python src/generate_subword.py "Write a Python function to add two numbers."
"""
from pathlib import Path
import argparse
import json
import torch
import sentencepiece as spm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GPT, GPTConfig

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CKPT_DIR = BASE_DIR / "checkpoints"
CONFIG_FILE = CKPT_DIR / "gpt_subword_config.json"


def load_model_and_tokenizer(prefix="gpt_subword", prefer_best=True, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg_d = json.loads((CKPT_DIR / f"{prefix}_config.json").read_text(encoding="utf-8"))
    cfg = GPTConfig.from_dict(cfg_d)

    ckpt = CKPT_DIR / f"{prefix}_best.pt"
    if not (prefer_best and ckpt.exists()):
        ckpt = CKPT_DIR / f"{prefix}.pt"
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model = GPT(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    sp = spm.SentencePieceProcessor()
    sp.load(str(PROCESSED_DIR / cfg_d["tokenizer_model"]))
    return model, sp, cfg_d, device, ckpt


def generate_one(model, sp, cfg_d, device, task, max_new_tokens=256,
                 temperature=0.8, top_k=40, top_p=0.95, repetition_penalty=1.15):
    prompt = f"### Instruction:\n{task}\n\n### Response:\n"
    ids = sp.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    y = model.generate(
        x, max_new_tokens=max_new_tokens, temperature=temperature,
        top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
        eos_id=cfg_d["eos_id"],
    )
    out = y[0].tolist()
    gen = out[len(ids):]                      # tokens the model produced
    if cfg_d["eos_id"] in gen:
        gen = gen[: gen.index(cfg_d["eos_id"])]  # stop at first EOS
    return sp.decode(ids + gen)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?",
                    default="Write a Python function to add two numbers.")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--rep", type=float, default=1.15)
    ap.add_argument("--sft", action="store_true", help="use the SFT checkpoint (gpt_sft)")
    args = ap.parse_args()

    model, sp, cfg_d, device, ckpt = load_model_and_tokenizer(
        prefix="gpt_sft" if args.sft else "gpt_subword")
    print(f"loaded {ckpt.name} on {device}\n")
    text = generate_one(model, sp, cfg_d, device, args.task,
                        max_new_tokens=args.tokens, temperature=args.temperature,
                        top_k=args.top_k, top_p=args.top_p,
                        repetition_penalty=args.rep)
    print(text)
