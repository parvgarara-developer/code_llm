"""
Rebuild the training data for the subword pipeline.

The raw JSONL/JSON sources are gone, but the 357 char-token shards in
data/processed/shards/ ARE the tokenized clean Python instruction/response
corpus produced by prepare_clean_code_dataset.py. Together with the char-level
vocab.json (itos) we can losslessly reconstruct the exact text, then:

  1. decode shards -> python_corpus.txt  (samples split by <|END_OF_SAMPLE|>)
  2. train a whitespace-preserving, byte-fallback SentencePiece BPE tokenizer
     (the old code_tokenizer.model destroyed newlines/indentation -> unusable)
  3. re-encode every sample to subword ids, append EOS after each sample, and
     stream them into a single uint16 memmap (tokens_subword.bin)
  4. write subword_meta.json describing the artifacts

Run:  venv/Scripts/python.exe src/prepare_subword_data.py
"""
from pathlib import Path
import json
import numpy as np
import torch
import sentencepiece as spm

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SHARD_DIR = PROCESSED_DIR / "shards"
CHAR_VOCAB_FILE = PROCESSED_DIR / "vocab.json"

CORPUS_TXT = PROCESSED_DIR / "python_corpus.txt"
TOK_PREFIX = PROCESSED_DIR / "code_bpe"          # -> code_bpe.model / .vocab
TOKENS_BIN = PROCESSED_DIR / "tokens_subword.bin"
META_FILE = PROCESSED_DIR / "subword_meta.json"

SEPARATOR = "<|END_OF_SAMPLE|>"
VOCAB_SIZE = 12000
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

# SentencePiece token ids fit in uint16 as long as vocab < 65536.
assert VOCAB_SIZE < 65536


def decode_shards_to_corpus():
    """Pass A: decode every char-token shard back to text and write the
    reconstructed corpus, with each training sample separated by SEPARATOR."""
    vocab = json.loads(CHAR_VOCAB_FILE.read_text(encoding="utf-8"))
    itos = {int(i): ch for i, ch in vocab["itos"].items()}

    shard_files = sorted(SHARD_DIR.glob("tokens_*.pt"))
    if not shard_files:
        raise FileNotFoundError(f"No shards found in {SHARD_DIR}")

    print(f"Decoding {len(shard_files)} shards -> {CORPUS_TXT.name}")
    total_chars = 0
    with CORPUS_TXT.open("w", encoding="utf-8", newline="") as out:
        for n, shard_path in enumerate(shard_files, 1):
            data = torch.load(shard_path, map_location="cpu", weights_only=True)
            text = "".join(itos[int(i)] for i in data.tolist())
            out.write(text)
            total_chars += len(text)
            if n % 25 == 0 or n == len(shard_files):
                print(f"  {n}/{len(shard_files)} shards  ({total_chars:,} chars)")
    print(f"Wrote {CORPUS_TXT} ({total_chars:,} chars)")


def train_tokenizer():
    """Pass B: train a code-safe SentencePiece BPE tokenizer.

    Critical flags vs the old broken tokenizer:
      remove_extra_whitespaces=False  -> keep indentation
      normalization_rule_name=identity -> don't fold whitespace/unicode
      byte_fallback=True               -> guarantee exact round-trip
      add_dummy_prefix=False           -> don't inject a leading space
    """
    print(f"Training SentencePiece tokenizer (vocab={VOCAB_SIZE}) ...")
    spm.SentencePieceTrainer.train(
        input=str(CORPUS_TXT),
        model_prefix=str(TOK_PREFIX),
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=1.0,
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        byte_fallback=True,
        add_dummy_prefix=False,
        unk_id=UNK_ID, bos_id=BOS_ID, eos_id=EOS_ID, pad_id=PAD_ID,
        unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>", pad_piece="<pad>",
        input_sentence_size=3_000_000,
        shuffle_input_sentence=True,
        num_threads=8,
        train_extremely_large_corpus=False,
    )
    print(f"Wrote {TOK_PREFIX}.model / {TOK_PREFIX}.vocab")


def tokenize_corpus():
    """Pass C: encode every sample to subword ids + EOS, streamed to uint16."""
    sp = spm.SentencePieceProcessor()
    sp.load(f"{TOK_PREFIX}.model")

    # Sanity: exact round-trip on a code snippet with indentation + newlines.
    probe = "def f(x):\n    if x:\n        return x\n    return 0\n"
    assert sp.decode(sp.encode(probe)) == probe, "tokenizer does not round-trip!"

    raw = CORPUS_TXT.read_text(encoding="utf-8")
    samples = [s.strip("\n") for s in raw.split(SEPARATOR)]
    samples = [s for s in samples if s.strip()]
    print(f"Encoding {len(samples):,} samples -> {TOKENS_BIN.name}")

    total_tokens = 0
    buf = []
    BATCH = 2000
    with TOKENS_BIN.open("wb") as f:
        for start in range(0, len(samples), BATCH):
            chunk = samples[start:start + BATCH]
            for ids in sp.encode(chunk):
                ids.append(EOS_ID)          # teach the model to stop
                buf.extend(ids)
            if len(buf) >= 4_000_000:
                arr = np.asarray(buf, dtype=np.uint16)
                arr.tofile(f)
                total_tokens += arr.size
                buf.clear()
            done = min(start + BATCH, len(samples))
            if done % 50000 < BATCH:
                print(f"  {done:,}/{len(samples):,} samples  ({total_tokens:,} tokens)")
        if buf:
            arr = np.asarray(buf, dtype=np.uint16)
            arr.tofile(f)
            total_tokens += arr.size

    meta = {
        "tokens_bin": TOKENS_BIN.name,
        "dtype": "uint16",
        "num_tokens": int(total_tokens),
        "num_samples": len(samples),
        "vocab_size": VOCAB_SIZE,
        "tokenizer_model": f"{TOK_PREFIX.name}.model",
        "eos_id": EOS_ID, "bos_id": BOS_ID, "unk_id": UNK_ID, "pad_id": PAD_ID,
    }
    META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nDone. {total_tokens:,} tokens over {len(samples):,} samples")
    print(f"Compression: {len(raw):,} chars -> {total_tokens:,} tokens "
          f"({len(raw)/max(total_tokens,1):.2f} chars/token)")
    print(f"Wrote {TOKENS_BIN} and {META_FILE}")


if __name__ == "__main__":
    decode_shards_to_corpus()
    train_tokenizer()
    tokenize_corpus()
