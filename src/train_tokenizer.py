from pathlib import Path
import sentencepiece as spm

INPUT_FILE = Path("data/processed/codesearchnet_python.txt")
MODEL_PREFIX = "data/processed/code_tokenizer"

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Missing corpus file: {INPUT_FILE}")

text = INPUT_FILE.read_text(encoding="utf-8").strip()
if not text:
    raise ValueError(f"Corpus file is empty: {INPUT_FILE}")

spm.SentencePieceTrainer.train(
    input=str(INPUT_FILE),
    model_prefix=MODEL_PREFIX,
    vocab_size=12000,
    model_type="bpe",
    character_coverage=1.0,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    max_sentence_length=20000,
    user_defined_symbols=["<|doc|>", "<|code|>", "<|endofsample|>"]
)

print("Tokenizer training complete.")