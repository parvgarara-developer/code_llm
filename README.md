# Code LLM — a GPT trained from scratch on Python code

A small **decoder-only GPT** (nanoGPT-style, implemented in raw PyTorch) trained
from scratch to generate Python code from a natural-language instruction. It runs
end-to-end on a single consumer GPU (developed on a 4 GB RTX 2050).

Given a prompt like *"Write a Python function to check if a string is a
palindrome,"* the trained model produces real, indented, self-terminating Python:

```python
def is_palindrome(s: str) -> bool:
    clean = ''.join(c.lower() for c in s if c.isalnum())
    return clean == clean[::-1]
```

> **Scope / expectations.** This is a ~15M-parameter model trained on ~600k
> functions. It reliably learns Python **syntax, structure, indentation, and when
> to stop**, and sometimes produces fully correct solutions — but it is not a
> general coding assistant. Many outputs are *plausible but not functionally
> correct*. See [Results & limitations](#results--limitations).

---

## Table of contents
- [Repository layout](#repository-layout)
- [Files NOT in this repo (and how to regenerate them)](#files-not-in-this-repo-and-how-to-regenerate-them)
- [Setup](#setup)
- [Datasets](#datasets)
- [The pipeline (recommended: subword)](#the-pipeline-recommended-subword)
- [Model architecture](#model-architecture)
- [Results & limitations](#results--limitations)
- [Legacy character-level pipeline](#legacy-character-level-pipeline)

---

## Repository layout

```
code_llm/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── model.py                     # single source of truth for the GPT architecture
│   ├── prepare_subword_data.py      # build corpus + train tokenizer + tokenize -> memmap
│   ├── train_subword.py             # training loop (AMP, cosine LR, memmap, resume)
│   ├── generate_subword.py          # generate from a single prompt
│   ├── evaluate_subword.py          # run the 10 benchmark prompts
│   │
│   ├── download_codesearchnet.py    # download + extract CodeSearchNet (Python)
│   ├── extract_codesearchnet.py     # (re-)extract the downloaded zip
│   ├── prepare_clean_code_dataset.py# raw JSON/JSONL -> cleaned char-token shards + vocab.json
│   │
│   └── ... (legacy char-level scripts, see below)
├── checkpoints/
│   ├── gpt_codesearchnet.pt         # small legacy checkpoint (in repo)
│   ├── gpt_opencode.pt              # small legacy checkpoint (in repo)
│   ├── gpt_opencode_config.json
│   └── gpt_subword_config.json      # config for the current model (weights not in repo)
└── output/
    ├── evaluation_outputs.txt       # OLD char-level model output (gibberish, for comparison)
    └── evaluation_subword.txt       # CURRENT subword model output
```

---

## Files NOT in this repo (and how to regenerate them)

Large binaries are intentionally **not** committed — GitHub rejects any file over
100 MB, and datasets/weights are regenerable artifacts. Everything below is
produced by the scripts in this repo; nothing here is lost by cloning.

| Path | Size | What it is | How to (re)create |
|---|---|---|---|
| `data/raw/` | ~1 GB+ | Raw datasets (CodeSearchNet + OpenCodeInstruct) | [Download](#datasets) |
| `data/processed/shards/` | ~5.7 GB | Char-token shards `tokens_*.pt` + `vocab.json` | `python src/prepare_clean_code_dataset.py` |
| `data/processed/python_corpus.txt` | ~714 MB | Reconstructed clean corpus | `python src/prepare_subword_data.py` |
| `data/processed/code_bpe.model` / `.vocab` | ~180 KB | Subword tokenizer (12k BPE) | `python src/prepare_subword_data.py` |
| `data/processed/tokens_subword.bin` | ~586 MB | 293M-token `uint16` memmap | `python src/prepare_subword_data.py` |
| `data/processed/subword_meta.json` | <1 KB | Token count, vocab size, EOS id, tokenizer path | `python src/prepare_subword_data.py` |
| `checkpoints/gpt_subword_best.pt` | ~175 MB | **Best trained model** (best val) | `python src/train_subword.py` |
| `checkpoints/gpt_subword.pt` | ~175 MB | Latest training checkpoint (with optimizer state, for resume) | `python src/train_subword.py` |
| `checkpoints/gpt_large_sharded.pt` | ~114 MB | Legacy large char-level checkpoint | `python src/train_large.py` |
| `venv/` | — | Python virtual environment | See [Setup](#setup) |

> Want the trained weights hosted anyway? Attach `gpt_subword_best.pt` to a
> **GitHub Release** (up to 2 GB per file) — the repo's 100 MB limit does not
> apply to release assets.

---

## Setup

Requires **Python 3.11** and (recommended) an NVIDIA GPU with CUDA. Training uses
bfloat16 automatic mixed precision, which needs an Ampere-or-newer GPU; it will
fall back to CPU but that is impractically slow for the full run.

```bash
python -m venv venv

# Windows (PowerShell)
venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt`: `torch`, `numpy`, `tqdm`, `sentencepiece`, `requests`.
Install the CUDA build of PyTorch for GPU training (see https://pytorch.org).

---

## Datasets

Two public Python datasets, combined into ~**607k** instruction/response samples
formatted as:

```
### Instruction:
<docstring or task>

### Response:
<python code>
```

### 1. CodeSearchNet (Python) — ~457k function/docstring pairs
Downloaded automatically from the public S3 mirror:

```bash
python src/download_codesearchnet.py
```

This fetches `python.zip` and extracts it to `data/raw/python_dataset/`. The
archive ships `.jsonl.gz` files; decompress them to `.jsonl` before the prep step:

```bash
# portable one-liner: gunzip every .jsonl.gz in place
python -c "import gzip,shutil,glob,os; [ (shutil.copyfileobj(gzip.open(p,'rb'), open(p[:-3],'wb')), os.remove(p)) for p in glob.glob('data/raw/python_dataset/**/*.jsonl.gz', recursive=True) ]"
```

Source: <https://github.com/github/CodeSearchNet> (dataset licensed per that repo).

### 2. OpenCodeInstruct — ~150k instruction/solution pairs
From NVIDIA's [`nvidia/OpenCodeInstruct`](https://huggingface.co/datasets/nvidia/OpenCodeInstruct)
on Hugging Face. Export it to a single JSON at
`data/raw/opencodeinstruct/opencodeinstruct.json` (a list of records, or
`{"data": [...]}`). The prep step reads fields such as
`instruction` / `input` / `prompt` and `output` / `response` / `code`:

```bash
python -c "from datasets import load_dataset; import json,os; os.makedirs('data/raw/opencodeinstruct',exist_ok=True); d=load_dataset('nvidia/OpenCodeInstruct', split='train'); json.dump([dict(r) for r in d], open('data/raw/opencodeinstruct/opencodeinstruct.json','w'))"
```

(Requires `pip install datasets`. Check the dataset card for license and exact
split/field names, then adjust if needed.)

Only Python samples are kept; a lightweight language check filters out non-Python
code.

---

## The pipeline (recommended: subword)

Run these in order from the repo root (with the venv active). On Windows,
substitute `venv\Scripts\python.exe` for `python` if the venv isn't activated.

### Step 1 — Build the cleaned dataset (char-token shards + vocab)
Parses `data/raw/**` into cleaned Python instruction/response samples and writes
char-token shards plus `vocab.json`:

```bash
python src/prepare_clean_code_dataset.py
```

### Step 2 — Build the subword corpus, tokenizer, and token stream
Reconstructs the text corpus from the shards, trains a **whitespace-preserving,
byte-fallback SentencePiece BPE tokenizer** (12k vocab), and tokenizes everything
into one `uint16` memmap with an EOS token between samples:

```bash
python src/prepare_subword_data.py
```

Produces `code_bpe.model`, `tokens_subword.bin` (~293M tokens), and
`subword_meta.json`.

> **Why retrain the tokenizer?** The old `code_tokenizer.model` was trained with
> SentencePiece's default whitespace normalization, which **destroys newlines and
> indentation** — fatal for Python. The new tokenizer round-trips code exactly.

### Step 3 — Train
Memmap data loader, real held-out validation split, bf16 AMP, AdamW with
weight-decay groups, warmup + cosine LR decay, gradient accumulation, gradient
clipping, and checkpoint/resume:

```bash
# full run (defaults: ~15M params, block 256, ~20k iters)
python src/train_subword.py --max-iters 20000

# resume an interrupted run
python src/train_subword.py --resume

# quick smoke test
python src/train_subword.py --max-iters 300 --eval-interval 100
```

Key flags: `--micro-batch`, `--grad-accum` (effective batch = product),
`--n-embd`, `--n-layer`, `--n-head`, `--block-size`, `--lr`, `--min-lr`.
Checkpoints are written to `checkpoints/gpt_subword.pt` (latest) and
`checkpoints/gpt_subword_best.pt` (best validation loss). The architecture is
saved to `checkpoints/gpt_subword_config.json` so generation reconstructs it
exactly.

### Step 4 — Generate
```bash
python src/generate_subword.py "Write a Python function to reverse a linked list."
```
Options: `--tokens`, `--temperature`, `--top-k`, `--top-p`, `--rep`
(repetition penalty). Generation stops at the EOS token.

### Step 5 — Evaluate
Runs 10 standard coding prompts and writes results to
`output/evaluation_subword.txt`:

```bash
python src/evaluate_subword.py
```

---

## Model architecture

Defined once in [`src/model.py`](src/model.py) and shared by training and
generation (so the config can never drift). Standard pre-LayerNorm transformer:

| Component | Detail |
|---|---|
| Attention | Fused QKV + `F.scaled_dot_product_attention` (memory-efficient, causal) |
| Blocks | Pre-LN, GELU MLP (4× expansion), residual connections |
| Embeddings | Learned token + positional; **weight tying** (token embedding = LM head) |
| Init | Normal(0, 0.02); residual projections scaled by 1/√(2·n_layer) |
| Optimizer | AdamW, decay on 2-D weights only, warmup + cosine schedule |

**Default config** (`GPTConfig`): `n_embd=384`, `n_head=6`, `n_layer=6`,
`block_size=256`, `vocab_size=12000`, `dropout=0.1` → **~15.35M parameters**.
Sized to fit ~2.6 GB VRAM with bf16. A 4 GB card can push to roughly
`--n-embd 512 --n-layer 8`.

---

## Results & limitations

The full 20k-iteration run (~4 h on an RTX 2050) reached a train loss of **~1.35**
(perplexity ≈ 3.9). Compare `output/evaluation_outputs.txt` (old char-level model)
against `output/evaluation_subword.txt` (current):

- **Before:** `def _self.____________________linen(self._______):` — gibberish.
- **After:** valid function signatures, correct indentation, docstrings, real
  control flow (`if`/`for`/`try`/`except`/`return`/`raise`), and clean stopping —
  with occasional fully-correct solutions.

**Honest caveats**
- Most outputs are **plausible but not always functionally correct** — expected
  for a 15M-parameter model; implementing algorithms from a spec needs far more
  scale and data than learning syntax does.
- The printed **validation loss is not trustworthy**: the held-out split is the
  contiguous tail of the token stream (all OpenCodeInstruct, heavily templated
  with near-duplicates), so it underestimates true difficulty. Judge by generated
  output, not that number. A random, deduplicated sample-level split is the fix.
- The model over-produces `self`/`cls` **methods** because CodeSearchNet is
  class-method-heavy; filtering to standalone functions would help.

**Next levers:** better data filtering (standalone functions, dedup, strip
JSON-wrapped samples), a trustworthy val split, and a larger/longer run — all
supported by the current pipeline.

---

## Legacy character-level pipeline

The original approach was **character-level** with a tiny 64-character context; it
produced gibberish and is superseded by the subword pipeline above. The scripts
are kept for reference:

- `prepare_char_tokens.py`, `prepare_codesearchnet_mixed.py`,
  `merge_corpora.py`, `tokenize_corpus.py`, `build_vocab.py`,
  `train_tokenizer.py` — legacy data/tokenizer prep.
- `model_gpt.py`, `model_gpt_large.py`, `train_gpt.py`, `train_large.py`,
  `generate.py`, `generate_large.py`, `evaluate_prompts.py` — legacy models,
  training, and inference (each defined its own architecture inline, which caused
  config drift — the reason `model.py` now centralizes it).
```
