"""One-shot generator for the versioned cipher-decode datasets.

Writes:
    data/cipher_train.jsonl   (3000 episodes, seed=1)
    data/cipher_val.jsonl     (200 episodes,  seed=2)
    data/cipher_test.jsonl    (500 episodes,  seed=3)

These files are committed to the repo. All Phase 2-7 conditions evaluate
against the same dataset to keep comparisons apples-to-apples.

Run:
    uv run python scripts/generate_cipher_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tasks.cipher_decode import generate_dataset, read_jsonl, write_jsonl  # noqa: E402

DATA_DIR = ROOT / "data"

SPLITS = [
    ("cipher_train.jsonl", 1, 3000),
    ("cipher_val.jsonl",   2, 200),
    ("cipher_test.jsonl",  3, 500),
]


def main() -> None:
    for filename, seed, n in SPLITS:
        path = DATA_DIR / filename
        episodes = generate_dataset(seed=seed, n=n)
        write_jsonl(path, episodes)
        # Verify round-trip
        loaded = list(read_jsonl(path))
        assert len(loaded) == n, f"{filename}: wrote {n} but loaded {len(loaded)}"
        assert loaded[0] == episodes[0], f"{filename}: round-trip mismatch on first episode"
        head = loaded[0]
        print(
            f"{filename}: n={n}  seed={seed}  "
            f"first_id={head['id']}  pt={head['plaintext']!r}  ct={head['ciphertext']!r}"
        )

    print(f"\nWrote datasets to {DATA_DIR}/")


if __name__ == "__main__":
    main()
