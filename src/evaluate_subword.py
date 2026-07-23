"""
Run the standard 10 coding prompts through the trained subword GPT and write
the results to output/evaluation_subword.txt (compare against the old
output/evaluation_outputs.txt to see the improvement).

  python src/evaluate_subword.py
"""
from pathlib import Path
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_subword import load_model_and_tokenizer, generate_one

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "evaluation_subword.txt"

PROMPTS = [
    "Write a Python function to add two numbers.",
    "Write a Python function to check if a string is a palindrome.",
    "Write a Python function to return the factorial of a number.",
    "Write a Python function to find the maximum element in a list.",
    "Write a Python function to count vowels in a string.",
    "Write a Python function to reverse a list without using built-in reverse().",
    "Write a Python function to remove duplicates from a list while keeping order.",
    "Write a Python function to check whether a number is prime.",
    "Write a Python function to merge two dictionaries.",
    "Write a Python function to compute the sum of digits of an integer.",
]

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", action="store_true", help="evaluate the SFT checkpoint (gpt_sft)")
    args = ap.parse_args()

    model, sp, cfg_d, device, ckpt = load_model_and_tokenizer(
        prefix="gpt_sft" if args.sft else "gpt_subword")
    out_path = OUTPUT_DIR / ("evaluation_sft.txt" if args.sft else "evaluation_subword.txt")
    print(f"loaded {ckpt.name} on {device}")

    blocks = []
    for i, task in enumerate(PROMPTS, 1):
        text = generate_one(model, sp, cfg_d, device, task,
                            max_new_tokens=256, temperature=0.7,
                            top_k=40, top_p=0.95, repetition_penalty=1.2)
        block = ("=" * 80 + f"\nPrompt {i}\nTask: {task}\n" + "-" * 80
                 + f"\n{text.strip()}\n\n")
        blocks.append(block)
        print(f"  [{i}/{len(PROMPTS)}] done")

    out_path.write_text("".join(blocks), encoding="utf-8")
    print(f"\nSaved -> {out_path}")
