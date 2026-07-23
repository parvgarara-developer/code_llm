"""
Supervised fine-tuning (SFT) with response-only loss.

Fine-tunes the pretrained base model (checkpoints/gpt_subword_best.pt) on the SFT
data from prepare_sft_data.py, computing loss ONLY on response tokens (mask==1).
This teaches "given the instruction, produce the code" instead of "model all the
text uniformly," which is what pushes the base model off its collapsed
`def write_*` template.

  python src/prepare_sft_data.py          # once, to build the data
  python src/train_sft.py                  # fine-tune from the pretrained model
  python src/evaluate_subword.py --sft     # see the result
"""
from pathlib import Path
import argparse
import json
import math
import sys
import time
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GPT, GPTConfig

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SFT_META = PROCESSED_DIR / "sft_meta.json"
CKPT_DIR = BASE_DIR / "checkpoints"
BASE_CONFIG = CKPT_DIR / "gpt_subword_config.json"
BASE_CKPT = CKPT_DIR / "gpt_subword_best.pt"
OUT_CKPT = CKPT_DIR / "gpt_sft.pt"
OUT_BEST = CKPT_DIR / "gpt_sft_best.pt"
OUT_CONFIG = CKPT_DIR / "gpt_sft_config.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-iters", type=int, default=4000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=100)
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)      # lower than pretraining
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--from-scratch", action="store_true",
                   help="don't load the pretrained base; train SFT from random init")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = device
    print("device:", device)
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    ctx = torch.autocast(device_type=device_type, dtype=amp_dtype) if device == "cuda" \
        else torch.autocast(device_type="cpu", enabled=False)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda" and not use_bf16))

    meta = json.loads(SFT_META.read_text(encoding="utf-8"))
    block_size = args.block_size

    def mm(name):
        return np.memmap(PROCESSED_DIR / meta[name], dtype=np.uint16 if "tok" in name else np.uint8, mode="r")
    tr_tok, tr_mask = mm("train_tok"), mm("train_mask")
    va_tok, va_mask = mm("val_tok"), mm("val_mask")
    print(f"train {len(tr_tok):,} tok ({meta['train_response_tokens']:,} scored) | val {len(va_tok):,}")

    def get_batch(split):
        tok, mask = (tr_tok, tr_mask) if split == "train" else (va_tok, va_mask)
        ix = torch.randint(len(tok) - block_size - 1, (args.micro_batch,))
        x = torch.stack([torch.from_numpy(tok[i:i + block_size].astype(np.int64)) for i in ix])
        yt = torch.stack([torch.from_numpy(tok[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
        ym = torch.stack([torch.from_numpy(mask[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
        y = torch.where(ym == 1, yt, torch.full_like(yt, -1))   # response-only loss
        if device == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)

    # architecture must match the base checkpoint to load its weights
    cfg_d = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    cfg = GPTConfig.from_dict({**cfg_d, "grad_checkpoint": args.grad_checkpoint,
                               "block_size": block_size})
    model = GPT(cfg).to(device)
    if not args.from_scratch:
        state = torch.load(BASE_CKPT, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        print(f"loaded pretrained weights from {BASE_CKPT.name}")
    print(f"params: {model.num_params()/1e6:.2f}M | effective batch {args.micro_batch*args.grad_accum}")

    optimizer = model.configure_optimizers(args.weight_decay, args.lr, (0.9, 0.95), device_type)
    OUT_CONFIG.write_text(json.dumps(
        {**{k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
         "tokenizer_model": meta["tokenizer_model"], "eos_id": meta["eos_id"]},
        indent=2), encoding="utf-8")

    def lr_at(it):
        if it < args.warmup:
            return args.lr * (it + 1) / args.warmup
        r = (it - args.warmup) / max(1, args.max_iters - args.warmup)
        return args.min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (args.lr - args.min_lr)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = []
            for _ in range(args.eval_iters):
                xb, yb = get_batch(split)
                if (yb != -1).sum() == 0:
                    continue
                with ctx:
                    _, loss = model(xb, yb)
                losses.append(loss.item())
            out[split] = sum(losses) / max(len(losses), 1)
        model.train()
        return out

    model.train()
    best_val = float("inf")
    t0 = time.time()
    for it in range(args.max_iters):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(it)
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            xb, yb = get_batch("train")
            with ctx:
                _, loss = model(xb, yb)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
        if args.grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            L = estimate_loss()
            print(f"iter {it:5d} | train {L['train']:.4f} | val {L['val']:.4f} "
                  f"| lr {lr_at(it):.2e} | {time.time()-t0:.0f}s")
            ck = {"model": model.state_dict(), "iter": it,
                  "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}}
            torch.save(ck, OUT_CKPT)
            if L["val"] < best_val:
                best_val = L["val"]
                torch.save(ck, OUT_BEST)

    print(f"done. best val (response-only) loss: {best_val:.4f}")
    print(f"checkpoint: {OUT_BEST}")


if __name__ == "__main__":
    main()
