"""
Build supervised fine-tuning (SFT) data with a response-only loss mask.

Pretraining scores every token, so the model spends capacity re-predicting the
instruction text and collapses to whatever code pattern is most frequent. SFT
fixes this by scoring ONLY the response tokens: the model is graded solely on the
code it should produce, given the instruction as context.

For each cleaned sample we emit two parallel streams:
  tokens_sft_{train,val}.bin  (uint16)  - prompt_ids + response_ids + EOS
  mask_sft_{train,val}.bin    (uint8)   - 0 over prompt tokens, 1 over response(+EOS)
Training ignores (sets target = -1) every position whose mask is 0.

Reuses the Phase 1 cleaning (unwrap JSON artifacts / drop non-parsing / dedup).

Run:  python src/prepare_sft_data.py
"""
from pathlib import Path
import argparse
import json
import random
import sys
import numpy as np
import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data_clean import (
    CORPUS_TXT, TOK_MODEL, PROCESSED_DIR, SEPARATOR, RESP_MARKER, EOS_ID,
    split_instruction_response, unwrap_json, parses, norm_key,
)

TRAIN_TOK = PROCESSED_DIR / "tokens_sft_train.bin"
TRAIN_MASK = PROCESSED_DIR / "mask_sft_train.bin"
VAL_TOK = PROCESSED_DIR / "tokens_sft_val.bin"
VAL_MASK = PROCESSED_DIR / "mask_sft_val.bin"
META_FILE = PROCESSED_DIR / "sft_meta.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--min-chars", type=int, default=24)
    ap.add_argument("--max-len", type=int, default=512, help="drop samples longer than this (tokens)")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    random.seed(args.seed)

    sp = spm.SentencePieceProcessor()
    sp.load(str(TOK_MODEL))

    raw_samples = CORPUS_TXT.read_text(encoding="utf-8").split(SEPARATOR)
    print(f"raw samples: {len(raw_samples):,}")

    seen = set()
    examples = []  # (prompt_ids, response_ids)
    stats = dict(raw=len(raw_samples), kept=0, dropped=0, dropped_long=0)

    for s in raw_samples:
        s = s.strip("\n")
        if not s:
            continue
        instr, code = split_instruction_response(s)
        if code is None:
            stats["dropped"] += 1
            continue
        code = unwrap_json(code)
        if len(code) < args.min_chars or not parses(code):
            stats["dropped"] += 1
            continue
        key = norm_key(code)
        if key in seen:
            stats["dropped"] += 1
            continue
        seen.add(key)

        prompt_ids = sp.encode(instr + RESP_MARKER)
        response_ids = sp.encode(code) + [EOS_ID]
        if len(prompt_ids) + len(response_ids) > args.max_len:
            stats["dropped_long"] += 1
            continue
        examples.append((prompt_ids, response_ids))
        stats["kept"] += 1

    print("stats:", {k: f"{v:,}" for k, v in stats.items()})

    random.shuffle(examples)
    n_val = int(len(examples) * args.val_frac)
    splits = {"val": examples[:n_val], "train": examples[n_val:]}
    print(f"split -> train {len(splits['train']):,} | val {len(splits['val']):,}")

    def write(tok_path, mask_path, exs):
        n_tok = n_resp = 0
        tbuf, mbuf = [], []
        with tok_path.open("wb") as tf, mask_path.open("wb") as mf:
            for p_ids, r_ids in exs:
                tbuf.extend(p_ids); tbuf.extend(r_ids)
                mbuf.extend([0] * len(p_ids)); mbuf.extend([1] * len(r_ids))
                n_tok += len(p_ids) + len(r_ids); n_resp += len(r_ids)
                if len(tbuf) >= 4_000_000:
                    np.asarray(tbuf, dtype=np.uint16).tofile(tf)
                    np.asarray(mbuf, dtype=np.uint8).tofile(mf)
                    tbuf.clear(); mbuf.clear()
            if tbuf:
                np.asarray(tbuf, dtype=np.uint16).tofile(tf)
                np.asarray(mbuf, dtype=np.uint8).tofile(mf)
        return n_tok, n_resp

    tr_tok, tr_resp = write(TRAIN_TOK, TRAIN_MASK, splits["train"])
    va_tok, va_resp = write(VAL_TOK, VAL_MASK, splits["val"])

    meta = {
        "train_tok": TRAIN_TOK.name, "train_mask": TRAIN_MASK.name,
        "val_tok": VAL_TOK.name, "val_mask": VAL_MASK.name,
        "train_tokens": tr_tok, "val_tokens": va_tok,
        "train_response_tokens": tr_resp, "val_response_tokens": va_resp,
        "train_examples": len(splits["train"]), "val_examples": len(splits["val"]),
        "vocab_size": sp.get_piece_size(), "eos_id": EOS_ID,
        "tokenizer_model": TOK_MODEL.name, "stats": stats,
    }
    META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"train tokens {tr_tok:,} ({tr_resp:,} scored) | val tokens {va_tok:,}")
    print(f"wrote {TRAIN_TOK.name}, {TRAIN_MASK.name}, {VAL_TOK.name}, {VAL_MASK.name}, {META_FILE.name}")


if __name__ == "__main__":
    main()
