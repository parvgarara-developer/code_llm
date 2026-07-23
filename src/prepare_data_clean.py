"""
Phase 1 data cleanup -> cleaner tokens + a trustworthy validation split.

Starts from the reconstructed corpus (data/processed/python_corpus.txt, samples
separated by <|END_OF_SAMPLE|>) and, per sample:
  1. unwrap JSON-artifact responses  ({"explanation":..., "code":...} -> the code)
  2. drop responses whose code does not parse (ast.parse)  -> removes broken code
  3. deduplicate on normalized response code  -> removes near-identical functions,
     which is what makes the old validation loss untrustworthy (train/val leakage)
Then shuffles and splits at the SAMPLE level into separate token streams:
  data/processed/tokens_train.bin   (uint16)
  data/processed/tokens_val.bin     (uint16, guaranteed disjoint from train)
so validation perplexity finally measures generalization.

Run:  python src/prepare_data_clean.py
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import random
import re
import numpy as np
import sentencepiece as spm

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CORPUS_TXT = PROCESSED_DIR / "python_corpus.txt"
TOK_MODEL = PROCESSED_DIR / "code_bpe.model"
TRAIN_BIN = PROCESSED_DIR / "tokens_train.bin"
VAL_BIN = PROCESSED_DIR / "tokens_val.bin"
META_FILE = PROCESSED_DIR / "clean_meta.json"

SEPARATOR = "<|END_OF_SAMPLE|>"
RESP_MARKER = "### Response:\n"
EOS_ID = 3


def split_instruction_response(sample: str):
    i = sample.find(RESP_MARKER)
    if i == -1:
        return None, None
    instr = sample[:i]
    code = sample[i + len(RESP_MARKER):]
    return instr, code.strip("\n")


def unwrap_json(code: str) -> str:
    """OpenCodeInstruct artifacts store the answer as a JSON dict. Pull the code."""
    s = code.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return code
    for loader in (json.loads, ast.literal_eval):
        try:
            obj = loader(s)
        except Exception:
            continue
        if isinstance(obj, dict):
            for key in ("fixed_code", "code", "solution", "response", "output", "answer"):
                v = obj.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
    return code


def parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False


_ws = re.compile(r"[ \t]+$", re.MULTILINE)


def norm_key(code: str) -> str:
    """Normalized form for dedup: strip trailing ws per line + blank edges."""
    c = _ws.sub("", code).strip("\n ")
    return hashlib.md5(c.encode("utf-8", "ignore")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--min-chars", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    random.seed(args.seed)

    if not CORPUS_TXT.exists():
        raise FileNotFoundError(f"Missing {CORPUS_TXT}. Run prepare_subword_data.py first.")

    raw = CORPUS_TXT.read_text(encoding="utf-8")
    raw_samples = raw.split(SEPARATOR)
    print(f"raw samples: {len(raw_samples):,}")

    stats = dict(raw=len(raw_samples), no_marker=0, unwrapped=0,
                 dropped_parse=0, dropped_short=0, dropped_dup=0, kept=0)
    seen = set()
    cleaned = []

    for s in raw_samples:
        s = s.strip("\n")
        if not s:
            continue
        instr, code = split_instruction_response(s)
        if code is None:
            stats["no_marker"] += 1
            continue

        new_code = unwrap_json(code)
        if new_code is not code and new_code != code:
            stats["unwrapped"] += 1
            code = new_code

        if len(code) < args.min_chars:
            stats["dropped_short"] += 1
            continue
        if not parses(code):
            stats["dropped_parse"] += 1
            continue

        key = norm_key(code)
        if key in seen:
            stats["dropped_dup"] += 1
            continue
        seen.add(key)

        cleaned.append(f"{instr}{RESP_MARKER}{code}")
        stats["kept"] += 1

    print("cleanup stats:")
    for k, v in stats.items():
        print(f"  {k:14s}: {v:,}")

    random.shuffle(cleaned)
    n_val = int(len(cleaned) * args.val_frac)
    val_samples = cleaned[:n_val]
    train_samples = cleaned[n_val:]
    print(f"\nsplit -> train {len(train_samples):,} | val {len(val_samples):,}")

    sp = spm.SentencePieceProcessor()
    sp.load(str(TOK_MODEL))

    def tokenize_to(path: Path, samples):
        total = 0
        buf = []
        with path.open("wb") as f:
            for start in range(0, len(samples), 2000):
                for ids in sp.encode(samples[start:start + 2000]):
                    ids.append(EOS_ID)
                    buf.extend(ids)
                if len(buf) >= 4_000_000:
                    a = np.asarray(buf, dtype=np.uint16); a.tofile(f); total += a.size; buf.clear()
            if buf:
                a = np.asarray(buf, dtype=np.uint16); a.tofile(f); total += a.size
        return total

    n_train_tok = tokenize_to(TRAIN_BIN, train_samples)
    n_val_tok = tokenize_to(VAL_BIN, val_samples)

    meta = {
        "train_bin": TRAIN_BIN.name, "val_bin": VAL_BIN.name, "dtype": "uint16",
        "train_tokens": int(n_train_tok), "val_tokens": int(n_val_tok),
        "train_samples": len(train_samples), "val_samples": len(val_samples),
        "vocab_size": sp.get_piece_size(), "eos_id": EOS_ID,
        "tokenizer_model": TOK_MODEL.name, "cleanup": stats,
    }
    META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\ntrain tokens: {n_train_tok:,} | val tokens: {n_val_tok:,}")
    print(f"kept {stats['kept']:,}/{stats['raw']:,} samples "
          f"({100*stats['kept']/stats['raw']:.1f}%)")
    print(f"wrote {TRAIN_BIN.name}, {VAL_BIN.name}, {META_FILE.name}")


if __name__ == "__main__":
    main()
