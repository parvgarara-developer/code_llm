"""
benchmark.py - measure functional correctness with pass@k on HumanEval / MBPP.

Generates candidate solutions with the trained subword model, executes each one
against the benchmark's hidden tests in a sandboxed subprocess, and reports the
unbiased pass@k estimator (Chen et al., 2021, "Evaluating LLMs Trained on Code").

This turns "is the code good?" into a number you can improve.

SAFETY: this executes model-generated code. Each candidate runs in a separate
process with a timeout and a reliability guard that neutralises the most
dangerous stdlib calls (os.system, shutil.rmtree, subprocess, file deletion...).
That is the standard approach for these benchmarks, but still run it on a machine
you are comfortable exposing to arbitrary Python. Use --oracle to validate the
harness itself: it runs the datasets' *canonical* solutions and should score
~100%.

Examples:
  python src/benchmark.py --oracle --limit 20                    # harness self-test
  python src/benchmark.py --dataset humaneval --limit 20 --n-samples 5   # quick
  python src/benchmark.py --dataset humaneval --n-samples 10     # full baseline
"""
from pathlib import Path
import argparse
import gzip
import json
import subprocess
import sys
import tempfile
import time
import numpy as np
import requests
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_subword import load_model_and_tokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
BENCH_DIR = BASE_DIR / "data" / "benchmarks"
OUT_DIR = BASE_DIR / "output"
BENCH_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

HUMANEVAL_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
MBPP_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"

# Prepended to every executed candidate: best-effort sandbox.
_GUARD = r"""
import os as _os, sys as _sys
try:
    import faulthandler; faulthandler.disable()
except Exception: pass
for _n in ("system","remove","unlink","rmdir","removedirs","rename","kill","killpg","chmod"):
    try: setattr(_os, _n, None)
    except Exception: pass
try:
    import shutil as _sh; _sh.rmtree = None; _sh.move = None
except Exception: pass
try:
    import subprocess as _sp; _sp.Popen = None; _sp.run = None; _sp.call = None
except Exception: pass
_os.environ["OMP_NUM_THREADS"] = "1"
try:
    import resource as _r; _r.setrlimit(_r.RLIMIT_CPU, (__CPU__, __CPU__))
except Exception: pass
from typing import *
import math, re, collections, itertools, functools, heapq, bisect, string
"""


def download(url: str, dest: Path):
    if dest.exists():
        return
    print(f"downloading {url}")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f"  -> {dest} ({len(r.content):,} bytes)")


def load_humaneval():
    gz = BENCH_DIR / "HumanEval.jsonl.gz"
    download(HUMANEVAL_URL, gz)
    with gzip.open(gz, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_mbpp():
    p = BENCH_DIR / "mbpp.jsonl"
    download(MBPP_URL, p)
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---- generation --------------------------------------------------------------

def generate(model, sp, cfg_d, device, prompt_text, max_new_tokens,
             temperature, top_k, top_p, rep, greedy):
    ids = sp.encode(prompt_text)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    y = model.generate(
        x, max_new_tokens=max_new_tokens,
        temperature=(1.0 if greedy else temperature),
        top_k=(1 if greedy else top_k),
        top_p=(1.0 if greedy else top_p),
        repetition_penalty=(1.0 if greedy else rep),
        eos_id=cfg_d["eos_id"],
    )
    out = y[0].tolist()[len(ids):]
    if cfg_d["eos_id"] in out:
        out = out[: out.index(cfg_d["eos_id"])]
    return sp.decode(out)


_STOPS = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n@", "\nassert ",
          "\n### ", "<|END", "\n\n\n"]


def truncate_completion(text: str) -> str:
    """Keep only the first function body a completion produces."""
    cut = len(text)
    for s in _STOPS:
        j = text.find(s)
        if j != -1:
            cut = min(cut, j)
    return text[:cut]


def extract_code_block(text: str) -> str:
    """For instruction-style output: strip anything after a clear boundary."""
    for s in ["\n### ", "<|END", "\n\n\n"]:
        j = text.find(s)
        if j != -1:
            text = text[:j]
    return text


# ---- execution ---------------------------------------------------------------

def run_program(program: str, timeout: int):
    try:
        p = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True, timeout=timeout, cwd=tempfile.gettempdir(),
        )
        if p.returncode == 0:
            return True, ""
        err = p.stderr.decode("utf-8", "ignore").strip().splitlines()
        return False, (err[-1][:160] if err else "")
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:  # noqa
        return False, str(e)[:160]


def guard(cpu_seconds: int) -> str:
    return _GUARD.replace("__CPU__", str(cpu_seconds))


# ---- pass@k ------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator: 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


# ---- per-dataset problem handling -------------------------------------------

def build_humaneval(prob, completion):
    program = prob["prompt"] + completion + "\n" + prob["test"] + \
        f"\ncheck({prob['entry_point']})\n"
    return program


def mbpp_entry_point(prob):
    # parse "assert func(...)" from the first test to learn the function name
    for t in prob.get("test_list", []):
        t = t.strip()
        if t.startswith("assert "):
            rest = t[len("assert "):].lstrip()
            name = rest.split("(")[0].strip()
            if name:
                return name
    return None


def build_mbpp(prob, code):
    setup = prob.get("test_setup_code", "") or ""
    tests = "\n".join(prob.get("test_list", []))
    return code + "\n" + setup + "\n" + tests + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="0 = all problems")
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--rep", type=float, default=1.1)
    ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("--ks", type=str, default="1,10", help="pass@k values, comma-sep")
    ap.add_argument("--oracle", action="store_true",
                    help="run canonical solutions to validate the harness (~100%)")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    problems = load_humaneval() if args.dataset == "humaneval" else load_mbpp()
    if args.limit:
        problems = problems[: args.limit]
    print(f"{args.dataset}: {len(problems)} problems | "
          f"{'ORACLE' if args.oracle else f'n_samples={args.n_samples}'}")

    if not args.oracle:
        model, sp, cfg_d, device, ckpt = load_model_and_tokenizer()
        print(f"model: {ckpt.name} on {device}")
    n_samples = 1 if args.oracle else args.n_samples
    g = guard(args.timeout)

    per_problem = []
    t0 = time.time()
    for pi, prob in enumerate(problems, 1):
        c = 0
        for _ in range(n_samples):
            if args.dataset == "humaneval":
                if args.oracle:
                    completion = prob["canonical_solution"]
                else:
                    completion = truncate_completion(generate(
                        model, sp, cfg_d, device, prob["prompt"],
                        args.max_new_tokens, args.temperature, args.top_k,
                        args.top_p, args.rep, greedy=(n_samples == 1)))
                program = g + build_humaneval(prob, completion)
            else:  # mbpp
                if args.oracle:
                    code = prob["code"]
                else:
                    hint = (prob.get("test_list") or [""])[0]
                    ptext = (f"### Instruction:\n{prob['text']}\n"
                             f"Your function must satisfy: {hint}\n\n### Response:\n")
                    code = extract_code_block(generate(
                        model, sp, cfg_d, device, ptext, args.max_new_tokens,
                        args.temperature, args.top_k, args.top_p, args.rep,
                        greedy=(n_samples == 1)))
                program = g + build_mbpp(prob, code)
            ok, _ = run_program(program, args.timeout)
            c += int(ok)
        per_problem.append((n_samples, c))
        if pi % 10 == 0 or pi == len(problems):
            solved = sum(1 for n, cc in per_problem if cc > 0)
            print(f"  [{pi}/{len(problems)}] solved>=1: {solved}  "
                  f"({time.time()-t0:.0f}s)")

    report = {"dataset": args.dataset, "problems": len(problems),
              "n_samples": n_samples, "oracle": args.oracle}
    print("\n" + "=" * 50)
    for k in ks:
        if k > n_samples and not args.oracle:
            continue
        score = float(np.mean([pass_at_k(n, c, k) for n, c in per_problem]))
        report[f"pass@{k}"] = score
        print(f"  pass@{k}: {score*100:.2f}%")
    report["solved_at_least_once"] = sum(1 for n, c in per_problem if c > 0)
    print(f"  solved at least once: {report['solved_at_least_once']}/{len(problems)}")
    print("=" * 50)

    tag = "oracle" if args.oracle else "model"
    out = OUT_DIR / f"benchmark_{args.dataset}_{tag}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
