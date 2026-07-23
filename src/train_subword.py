"""
Train the subword GPT on the memmapped token stream.

Fixes every training-loop problem from train_large.py:
  - no more per-step torch.load: one np.memmap, random windows, ~0 I/O
  - real held-out validation split (train/val no longer overlap)
  - AMP (bf16 on Ampere, else fp16+GradScaler) so a ~15M-param model fits 4GB
  - AdamW with decay/no-decay groups, warmup + cosine LR decay, grad clipping
  - gradient accumulation for a larger effective batch
  - checkpoint + resume, and the model config is saved for generation

Examples:
  smoke test:  python src/train_subword.py --max-iters 300 --eval-interval 100
  full run:    python src/train_subword.py               (background this one)
  resume:      python src/train_subword.py --resume
"""
from pathlib import Path
import argparse
import json
import math
import time
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GPT, GPTConfig

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
TOKENS_BIN = PROCESSED_DIR / "tokens_subword.bin"
META_FILE = PROCESSED_DIR / "subword_meta.json"
CLEAN_META = PROCESSED_DIR / "clean_meta.json"          # Phase 1 clean split
TRAIN_BIN = PROCESSED_DIR / "tokens_train.bin"
VAL_BIN = PROCESSED_DIR / "tokens_val.bin"
CKPT_DIR = BASE_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_FILE = CKPT_DIR / "gpt_subword.pt"
CONFIG_FILE = CKPT_DIR / "gpt_subword_config.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-iters", type=int, default=12000)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-iters", type=int, default=100)
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--no-clean", action="store_true",
                   help="ignore the Phase 1 clean split, use the old positional split")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device == "cuda" else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        use_bf16 = torch.cuda.is_bf16_supported()
    else:
        use_bf16 = False
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    ctx = (torch.autocast(device_type=device_type, dtype=amp_dtype)
           if device == "cuda" else torch.autocast(device_type="cpu", enabled=False))
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and not use_bf16))
    print(f"AMP dtype: {amp_dtype}  (GradScaler {'on' if scaler.is_enabled() else 'off'})")

    # Prefer the Phase 1 clean split (separate, guaranteed-disjoint train/val
    # bins) when available; otherwise fall back to the single-file positional
    # split (which leaks and gives an untrustworthy val loss).
    if not args.no_clean and CLEAN_META.exists() and TRAIN_BIN.exists() and VAL_BIN.exists():
        meta = json.loads(CLEAN_META.read_text(encoding="utf-8"))
        vocab_size = meta["vocab_size"]
        train_data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
        val_data = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")
        print(f"[clean split] train: {len(train_data):,}  val: {len(val_data):,} tokens (disjoint)")
    else:
        meta = json.loads(META_FILE.read_text(encoding="utf-8"))
        vocab_size = meta["vocab_size"]
        data = np.memmap(TOKENS_BIN, dtype=np.uint16, mode="r")
        n = len(data)
        n_val = int(n * args.val_frac)
        train_data = data[: n - n_val]
        val_data = data[n - n_val:]
        print(f"[positional split] tokens: {n:,}  train: {len(train_data):,}  val: {len(val_data):,}")

    block_size = args.block_size

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - block_size - 1, (args.micro_batch,))
        x = torch.stack([torch.from_numpy(d[i:i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(d[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
        if device == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)

    cfg = GPTConfig(
        vocab_size=vocab_size, block_size=block_size,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(device)
    print(f"params: {model.num_params()/1e6:.2f}M  "
          f"effective batch: {args.micro_batch * args.grad_accum}")

    optimizer = model.configure_optimizers(
        args.weight_decay, args.lr, (0.9, 0.95), device_type)

    start_iter = 0
    best_val = float("inf")
    if args.resume and CKPT_FILE.exists():
        ck = torch.load(CKPT_FILE, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_iter = ck["iter"] + 1
        best_val = ck.get("best_val", best_val)
        print(f"resumed from iter {start_iter}  (best_val {best_val:.4f})")

    # save the config so generation reconstructs the exact architecture
    CONFIG_FILE.write_text(json.dumps(
        {**{k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
         "tokenizer_model": meta["tokenizer_model"], "eos_id": meta["eos_id"]},
        indent=2), encoding="utf-8")

    def lr_at(it):
        if it < args.warmup:
            return args.lr * (it + 1) / args.warmup
        if it >= args.max_iters:
            return args.min_lr
        ratio = (it - args.warmup) / max(1, args.max_iters - args.warmup)
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return args.min_lr + coeff * (args.lr - args.min_lr)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(args.eval_iters)
            for i in range(args.eval_iters):
                xb, yb = get_batch(split)
                with ctx:
                    _, loss = model(xb, yb)
                losses[i] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    model.train()
    t0 = time.time()
    for it in range(start_iter, args.max_iters):
        lr = lr_at(it)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        for micro in range(args.grad_accum):
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
            losses = estimate_loss()
            dt = time.time() - t0
            print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"| lr {lr:.2e} | {dt:.1f}s")
            t0 = time.time()
            ck = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                  "iter": it, "best_val": min(best_val, losses["val"]),
                  "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}}
            torch.save(ck, CKPT_FILE)
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(ck, CKPT_DIR / "gpt_subword_best.pt")

    print(f"done. best val loss: {best_val:.4f}")
    print(f"checkpoint: {CKPT_FILE}")


if __name__ == "__main__":
    main()
