# Code LLM — a GPT trained from scratch on Python code

A small **decoder-only GPT** (nanoGPT-style, implemented in raw PyTorch — no
Hugging Face, no pretrained weights) trained from scratch to generate Python from
a natural-language instruction. Data prep, tokenizer, training, inference,
benchmarking and a web UI all run end-to-end on a single consumer GPU
(developed on a 4 GB RTX 2050).

Here is a real, unedited sample — prompted with *"Write a Python function to
check whether a number is prime"* (verbatim from
[`output/evaluation_subword.txt`](output/evaluation_subword.txt)):

```python
def _write_python(self, func):
        """ Write a Python function to check whether a number is prime."""
        if isinstance(func, int):
            if not self._readable:
                raise NotImplementedError("'%s' does not support %r" % (func,))
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning('Could not write python function.')
```

That is the honest picture, and it is the point of this repo: the model has
thoroughly learned Python **form** — valid syntax, indentation, docstrings,
exception handling, and when to stop — while learning the **task** not at all.
It scores **pass@1 = 0% on HumanEval**, measured with an oracle-verified harness
rather than eyeballed. That is the expected outcome at ~15M parameters, and
[Results](#results--what-was-actually-measured) breaks down exactly why.

---

## Table of contents
- [Quick start (web UI)](#quick-start-web-ui)
- [Repository layout](#repository-layout)
- [Files NOT in this repo (and how to regenerate them)](#files-not-in-this-repo-and-how-to-regenerate-them)
- [Setup](#setup)
- [Datasets](#datasets)
- [The pipeline](#the-pipeline)
- [Web UI](#web-ui)
- [Model architecture](#model-architecture)
- [Results & what was actually measured](#results--what-was-actually-measured)
- [Legacy character-level pipeline](#legacy-character-level-pipeline)

---

## Quick start (web UI)

If the model artifacts are already in place (`checkpoints/` + `data/processed/`):

```bash
venv/Scripts/python.exe ui_server.py 5000
```

Open **http://localhost:5000** — the backend serves the UI and runs inference on
the GPU. To build the artifacts from scratch instead, follow
[The pipeline](#the-pipeline).

---

## Repository layout

```
code_llm/
├── README.md
├── requirements.txt
├── ui_server.py                     # UI backend + inference API (stdlib http.server)
├── ui/
│   ├── index.html                   # single-page playground
│   ├── index.css                    # neo-brutalist design system
│   └── index.js                     # streaming, highlighting, visualizers
├── src/
│   │   # --- current subword pipeline ---
│   ├── model.py                     # single source of truth for the GPT architecture
│   ├── prepare_subword_data.py      # corpus -> BPE tokenizer -> uint16 memmap
│   ├── prepare_data_clean.py        # dedup/parse-filter + leak-free train/val split
│   ├── train_subword.py             # pretraining (AMP, cosine LR, memmap, resume)
│   ├── prepare_sft_data.py          # instruction data + response-only loss mask
│   ├── train_sft.py                 # instruction tuning (SFT)
│   ├── generate_subword.py          # generate from a single prompt
│   ├── evaluate_subword.py          # run the 10 standard prompts
│   ├── benchmark.py                 # HumanEval / MBPP pass@k with sandboxed execution
│   │
│   │   # --- dataset acquisition ---
│   ├── download_codesearchnet.py    # download + extract CodeSearchNet (Python)
│   ├── extract_codesearchnet.py     # (re-)extract the downloaded zip
│   ├── prepare_clean_code_dataset.py# raw JSON/JSONL -> char-token shards + vocab.json
│   │
│   └── ...                          # legacy char-level scripts (see below)
├── checkpoints/
│   ├── gpt_subword_config.json      # config for the pretrained model
│   ├── gpt_sft_config.json          # config for the instruction-tuned model
│   ├── gpt_opencode.pt              # small legacy checkpoint (in repo)
│   ├── gpt_codesearchnet.pt         # small legacy checkpoint (in repo)
│   └── gpt_opencode_config.json
└── output/
    ├── evaluation_outputs.txt       # OLD char-level output (gibberish, for contrast)
    ├── evaluation_subword.txt       # current model output
    ├── benchmark_humaneval_oracle.json  # harness self-test (100%)
    └── benchmark_humaneval_model.json   # measured model score
```

---

## Files NOT in this repo (and how to regenerate them)

Large binaries are intentionally **not** committed — GitHub rejects any file over
100 MB, and datasets/weights are regenerable. Everything below is produced by the
scripts here; nothing is lost by cloning.

| Path | Size | What it is | How to (re)create |
|---|---|---|---|
| `data/raw/` | ~1 GB+ | Raw datasets (CodeSearchNet + OpenCodeInstruct) | [Download](#datasets) |
| `data/processed/shards/` | ~5.7 GB | Char-token shards `tokens_*.pt` + `vocab.json` | `prepare_clean_code_dataset.py` |
| `data/processed/python_corpus.txt` | ~681 MB | Reconstructed clean corpus (714M chars) | `prepare_subword_data.py` |
| `data/processed/code_bpe.model` / `.vocab` | ~180 KB | Subword tokenizer (12k BPE) | `prepare_subword_data.py` |
| `data/processed/tokens_subword.bin` | ~558 MB | 293M-token `uint16` memmap | `prepare_subword_data.py` |
| `data/processed/tokens_train.bin` / `tokens_val.bin` | ~517 MB | Cleaned, **disjoint** train/val streams | `prepare_data_clean.py` |
| `data/processed/tokens_sft_*.bin` / `mask_sft_*.bin` | ~220 MB | SFT tokens + response-loss masks | `prepare_sft_data.py` |
| `checkpoints/gpt_subword_best.pt` | ~175 MB | **Pretrained model** (best val) | `train_subword.py` |
| `checkpoints/gpt_subword.pt` | ~175 MB | Latest checkpoint w/ optimizer state (resume) | `train_subword.py` |
| `checkpoints/gpt_sft_best.pt` | ~58 MB | **Instruction-tuned model** | `train_sft.py` |
| `checkpoints/gpt_large_sharded.pt` | ~114 MB | Legacy large char-level checkpoint | `train_large.py` |
| `venv/` | — | Python virtual environment | See [Setup](#setup) |

> Want the weights hosted anyway? Attach `gpt_subword_best.pt` to a **GitHub
> Release** (2 GB per file) — the repo's 100 MB limit doesn't apply to release
> assets.

---

## Setup

Requires **Python 3.11** and, realistically, an NVIDIA GPU. Training uses bfloat16
mixed precision (Ampere or newer); it falls back to CPU but that is impractically
slow for a full run.

```bash
python -m venv venv

# Windows (PowerShell)
venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt`: `torch`, `numpy`, `tqdm`, `sentencepiece`, `requests`.
Install the CUDA build of PyTorch for GPU training (see <https://pytorch.org>).

---

## Datasets

Two public Python datasets, normalised into ~**607k** instruction/response
samples of the form:

```
### Instruction:
<docstring or task>

### Response:
<python code>
```

### 1. CodeSearchNet (Python) — ~457k function/docstring pairs

```bash
python src/download_codesearchnet.py
```

Fetches `python.zip` from the public S3 mirror and extracts to
`data/raw/python_dataset/`. The archive ships `.jsonl.gz`; decompress before the
prep step:

```bash
python -c "import gzip,shutil,glob,os; [ (shutil.copyfileobj(gzip.open(p,'rb'), open(p[:-3],'wb')), os.remove(p)) for p in glob.glob('data/raw/python_dataset/**/*.jsonl.gz', recursive=True) ]"
```

Source: <https://github.com/github/CodeSearchNet> (licensed per that repo).

### 2. OpenCodeInstruct — ~150k instruction/solution pairs

From NVIDIA's [`nvidia/OpenCodeInstruct`](https://huggingface.co/datasets/nvidia/OpenCodeInstruct).
Export to `data/raw/opencodeinstruct/opencodeinstruct.json` (a list of records,
or `{"data": [...]}`):

```bash
python -c "from datasets import load_dataset; import json,os; os.makedirs('data/raw/opencodeinstruct',exist_ok=True); d=load_dataset('nvidia/OpenCodeInstruct', split='train'); json.dump([dict(r) for r in d], open('data/raw/opencodeinstruct/opencodeinstruct.json','w'))"
```

(Requires `pip install datasets`; check the dataset card for license and field
names.) Only Python samples are kept — a language check filters the rest.

---

## The pipeline

Run from the repo root with the venv active. On Windows substitute
`venv\Scripts\python.exe` for `python` if the venv isn't activated.

### Step 1 — Build the cleaned dataset

```bash
python src/prepare_clean_code_dataset.py
```

Parses `data/raw/**` into instruction/response samples and writes char-token
shards plus `vocab.json`.

### Step 2 — Tokenizer + token stream

```bash
python src/prepare_subword_data.py
```

Reconstructs the text corpus from the shards, trains a **whitespace-preserving,
byte-fallback SentencePiece BPE tokenizer** (12k vocab), and tokenizes everything
into one `uint16` memmap with EOS between samples → `code_bpe.model`,
`tokens_subword.bin` (~293M tokens), `subword_meta.json`.

> **Why retrain the tokenizer?** The original `code_tokenizer.model` used
> SentencePiece's default whitespace normalisation, which **destroys newlines and
> indentation** — fatal for Python. The replacement round-trips code byte-exactly
> (`remove_extra_whitespaces=False`, `byte_fallback=True`, identity normaliser).

### Step 3 — Clean the data + build a leak-free split *(recommended)*

```bash
python src/prepare_data_clean.py
```

Unwraps JSON-artifact responses, drops code that fails `ast.parse`, deduplicates
near-identical functions, then writes **disjoint** `tokens_train.bin` /
`tokens_val.bin`. `train_subword.py` picks these up automatically when present.

Measured effect on the corpus:

| | Count |
|---|---|
| Raw samples | 607,544 |
| JSON-wrapped responses unwrapped | 150,082 |
| Dropped — didn't parse | 79,385 |
| Dropped — duplicates | 75,199 |
| **Kept** | **452,959 (74.6%)** |

Without this step the validation loss is meaningless: the split was a contiguous
tail of a corpus full of near-duplicates, so it reported **0.039** while train sat
at 1.35 — memorisation, not generalisation.

### Step 4 — Pretrain

```bash
# full run (~15M params, block 256, ~4 h on an RTX 2050)
python src/train_subword.py --max-iters 20000

python src/train_subword.py --resume                        # resume
python src/train_subword.py --max-iters 300 --eval-interval 100   # smoke test
```

Memmap loader, bf16 AMP, AdamW with weight-decay groups, warmup + cosine decay,
gradient accumulation, gradient clipping, checkpoint/resume. Key flags:
`--micro-batch`, `--grad-accum` (effective batch = product), `--n-embd`,
`--n-layer`, `--n-head`, `--block-size`, `--lr`, `--min-lr`, `--no-clean`.

Writes `checkpoints/gpt_subword.pt` (latest), `gpt_subword_best.pt` (best val),
and `gpt_subword_config.json` so inference rebuilds the architecture exactly.

### Step 5 — Instruction-tune (SFT)

```bash
python src/prepare_sft_data.py
python src/train_sft.py --max-iters 4000
```

Pretraining scores *every* token, so the model spends capacity re-predicting the
instruction and drifts toward whatever code pattern is most frequent. SFT scores
**only the response tokens** (prompt positions are set to `ignore_index`), which
targets instruction-following directly. `prepare_sft_data.py` emits parallel
token and mask streams; `train_sft.py` fine-tunes from `gpt_subword_best.pt` at a
lower LR and writes `gpt_sft_best.pt`.

> **Status:** the SFT stage is wired end-to-end and produces a working
> checkpoint, but the run in this repo was stopped at iteration 2000 of 4000, so
> it is **not a completed fine-tune**. Every number in
> [Results](#results--what-was-actually-measured) is from the *pretrained* model.

### Step 6 — Generate

```bash
python src/generate_subword.py "Write a Python function to reverse a linked list."
python src/generate_subword.py "..." --sft        # use the instruction-tuned model
```

Options: `--tokens`, `--temperature`, `--top-k`, `--top-p`, `--rep` (repetition
penalty). Generation stops at EOS.

### Step 7 — Evaluate

```bash
python src/evaluate_subword.py           # -> output/evaluation_subword.txt
python src/evaluate_subword.py --sft     # -> output/evaluation_sft.txt
```

### Step 8 — Benchmark correctness (pass@k)

Eyeballing output only goes so far. [`src/benchmark.py`](src/benchmark.py)
measures *functional* correctness on **HumanEval** (and MBPP): it generates
candidates, executes them against the benchmark's hidden tests in a sandboxed
subprocess, and reports the unbiased **pass@k** estimator (Chen et al., 2021).

```bash
python src/benchmark.py --oracle                             # validate the harness
python src/benchmark.py --dataset humaneval --n-samples 10   # score the model
```

`--oracle` runs the datasets' *canonical* solutions and must score ~100% — it
proves the executor and scoring are correct independently of the model. On this
repo it scores **164/164 = 100%**.

> ⚠️ This executes model-generated code. Each candidate runs in a separate
> process with a timeout and a guard that neutralises the most destructive stdlib
> calls, which is the standard approach for these benchmarks — but still run it
> somewhere you're comfortable executing arbitrary Python.

---

## Web UI

A single-page playground served by [`ui_server.py`](ui_server.py) — a dependency-free
`http.server` backend that serves the static UI *and* runs inference in-process.

```bash
venv/Scripts/python.exe ui_server.py 5000     # then open http://localhost:5000
```

Features: streaming token-by-token output with Python syntax highlighting,
pause / step / stop controls, live inference stats (time, tokens, tok/s, device),
a causal self-attention grid, a BPE token-shard view, a pipeline log console, and
a neo-brutalist theme with light/dark modes.

Models are **cached after first load** (first request ~7 s to read the
checkpoint, subsequent requests ~0.5 s).

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/status` | GET | torch/CUDA availability, GPU name, which models exist |
| `/api/models` | GET | model registry |
| `/api/generate` | POST | run inference |

```bash
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt_subword","prompt":"Write a function to add two numbers.","max_tokens":80}'
```

`model` accepts `gpt_subword` (default), `gpt_sft`, or the legacy character-level
`gpt_opencode` / `gpt_codesearchnet`. Other fields: `temperature`, `top_k`,
`top_p`, `max_tokens`, `repetition_penalty`.

---

## Model architecture

Defined once in [`src/model.py`](src/model.py) and shared by training, inference
and the UI, so the config can never drift:

| Component | Detail |
|---|---|
| Attention | Fused QKV + `F.scaled_dot_product_attention` (memory-efficient, causal) |
| Blocks | Pre-LN, GELU MLP (4× expansion), residual connections |
| Embeddings | Learned token + positional; **weight tying** (token embedding = LM head) |
| Init | Normal(0, 0.02); residual projections scaled by 1/√(2·n_layer) |
| Optimizer | AdamW, decay on 2-D weights only, warmup + cosine schedule |
| Memory | Optional gradient checkpointing (`grad_checkpoint`) for bigger models on small GPUs |

**Default config** (`GPTConfig`): `n_embd=384`, `n_head=6`, `n_layer=6`,
`block_size=256`, `vocab_size=12000`, `dropout=0.1` → **~15.35M parameters**,
~2.6 GB VRAM in bf16. A 4 GB card can push to roughly `--n-embd 512 --n-layer 8`.

Gradient checkpointing lives in `GPTConfig(grad_checkpoint=True)` and is exposed
as `--grad-checkpoint` on `train_sft.py`. `train_subword.py` does **not** expose
the flag yet — pass it through there if you want to pretrain a larger model on a
small GPU.

---

## Results & what was actually measured

### Training

| Run | Train loss | Val loss | Verdict |
|---|---|---|---|
| Pretrain, raw data | 1.35 | **0.039** | Val 34× *below* train — leakage, not learning |
| Pretrain, cleaned data | 1.42 | **1.41** | train ≈ val — honest, no overfit (ppl ≈ 4.1) |

The second row is the real one. Fixing the split made the number *look* worse and
*be* trustworthy.

### Generation quality

| Stage | Representative output |
|---|---|
| Char-level (original) | `def _self.____________________linen(self._______):` |
| Subword, raw data | Valid Python; occasional fully-correct solutions; JSON artifacts leaked through |
| Subword, cleaned data | Valid Python, no artifacts, correct indentation, docstrings, real control flow |

### Functional correctness

| Benchmark | Score |
|---|---|
| HumanEval, canonical solutions (`--oracle`) | **100%** (164/164) — harness verified |
| HumanEval, this model | **pass@1 = 0%**, pass@10 = 0% (0/164) |

### What this demonstrates

The model learned Python **form** thoroughly and **task-following** not at all.
Ask for a palindrome check, a prime test, or a digit sum and it tends to emit the
same high-frequency CodeSearchNet shapes — a `def write_*(...)` wrapper or a
decorator boilerplate — with the instruction echoed into the docstring.

That is the honest, expected outcome at this scale, and it isolates *why*:

- **Scale.** ~15M parameters and 293M tokens. Genuinely useful open code models
  are ~1.5B+ parameters trained on trillions of tokens — roughly 100× the
  parameters and 19,000× the data. No amount of cleanup closes that.
- **Objective.** A base LM is trained to continue text, not to satisfy a request.
  Instruction-following comes from an SFT/preference stage on top of scale.
- **Data bias.** CodeSearchNet is class-method-heavy, so the model over-produces
  `self`/`cls` methods. Filtering to standalone functions would help.

### Next levers, in order of impact

1. **Scale up** — 100–350M params on a rented/cloud GPU is where instruction
   following starts to appear. The pipeline already supports it.
2. **Finish instruction tuning** — full SFT run, then preference tuning (DPO).
3. **Modernise the architecture** — RoPE, RMSNorm, SwiGLU (the Llama/Qwen recipe)
   for better quality per parameter.
4. **Execution-filtered data** — keep only samples whose code passes its own
   tests, and sample-then-test at inference.

---

## Legacy character-level pipeline

The original approach was character-level with a 64-character context. It
produced gibberish and is superseded by the subword pipeline above; the scripts
are kept for reference:

- `prepare_char_tokens.py`, `prepare_codesearchnet_mixed.py`, `merge_corpora.py`,
  `tokenize_corpus.py`, `build_vocab.py`, `train_tokenizer.py` — legacy data and
  tokenizer prep.
- `model_gpt.py`, `model_gpt_large.py`, `train_gpt.py`, `train_large.py`,
  `generate.py`, `generate_large.py`, `evaluate_prompts.py` — legacy models,
  training and inference. Each defined its architecture inline, which caused the
  config drift that `model.py` now prevents.
