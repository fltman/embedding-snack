"""One-shot generator for the versioned cipher-decode datasets.

Writes both v1 (full a-z, 5-15 char plaintext) and v2 (a-l, 3-6 char plaintext)
splits. All committed to the repo so any condition can be reproduced exactly.

v1 (kept for historical comparability — the marginal-4B run):
    data/cipher_train.jsonl   3000 ep, seed=1, a-z, 5-15 chars
    data/cipher_val.jsonl      200 ep, seed=2
    data/cipher_test.jsonl     500 ep, seed=3

v2 (current default — chosen after 4B was marginal on 26-letter version):
    data/cipher_v2_train.jsonl 3000 ep, seed=11, a-l (12 letters), 3-6 chars
    data/cipher_v2_val.jsonl    200 ep, seed=12
    data/cipher_v2_test.jsonl   500 ep, seed=13

Run:
    uv run python scripts/generate_cipher_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tasks.cipher_decode import (  # noqa: E402
    ALPHABET,
    ALPHABET_12,
    generate_dataset,
    read_jsonl,
    write_jsonl,
)

DATA_DIR = ROOT / "data"

SPLITS = [
    # filename, seed, n, alphabet, min_len, max_len
    ("cipher_train.jsonl",     1,  3000, ALPHABET,    5, 15),
    ("cipher_val.jsonl",       2,   200, ALPHABET,    5, 15),
    ("cipher_test.jsonl",      3,   500, ALPHABET,    5, 15),
    ("cipher_v2_train.jsonl", 11,  3000, ALPHABET_12, 3,  6),
    ("cipher_v2_val.jsonl",   12,   200, ALPHABET_12, 3,  6),
    ("cipher_v2_test.jsonl",  13,   500, ALPHABET_12, 3,  6),
]


def main() -> None:
    for filename, seed, n, alphabet, min_len, max_len in SPLITS:
        path = DATA_DIR / filename
        if path.exists():
            existing = list(read_jsonl(path))
            if len(existing) == n:
                print(f"{filename}: already present (n={n}); skipping")
                continue
        episodes = generate_dataset(seed=seed, n=n, alphabet=alphabet,
                                    min_len=min_len, max_len=max_len)
        write_jsonl(path, episodes)
        loaded = list(read_jsonl(path))
        assert len(loaded) == n, f"{filename}: wrote {n} but loaded {len(loaded)}"
        assert loaded[0] == episodes[0], f"{filename}: round-trip mismatch on first episode"
        head = loaded[0]
        print(
            f"{filename}: n={n}  seed={seed}  alphabet={alphabet[:6]}{'...' if len(alphabet)>6 else ''}({len(alphabet)})  "
            f"pt_len=[{min_len},{max_len}]  first: pt={head['plaintext']!r} ct={head['ciphertext']!r}"
        )

    print(f"\nWrote datasets to {DATA_DIR}/")


if __name__ == "__main__":
    main()
