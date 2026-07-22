import sentencepiece as spm
from pathlib import Path

corpus = "data/processed/corpus.txt"
model_prefix = "data/processed/code_tokenizer"

spm.SentencePieceTrainer.train(
    input=corpus,
    model_prefix=model_prefix,
    vocab_size=8000,
    model_type="bpe",
    character_coverage=1.0,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3
)

print("Tokenizer trained")