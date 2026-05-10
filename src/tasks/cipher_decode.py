"""Cooperative cipher-decode task.

Episode: random monoalphabetic substitution cipher + random plaintext.
Player A sees ciphertext only. Player B sees the cipher key only.
Goal: B emits the decoded plaintext.

JSONL schema (one episode per line):
    {
      "id": int,
      "plaintext": str,        # k chars from `alphabet`, k in [min_len, max_len]
      "ciphertext": str,       # plaintext mapped through key
      "key": str               # permutation of `alphabet`; key[i] is image of alphabet[i]
    }

The alphabet length is implicit from the key length, so we don't store it.
v1 datasets (cipher_*.jsonl) use the full 26-letter alphabet, plaintexts 5-15.
v2 datasets (cipher_v2_*.jsonl) use alphabet 'a'-'l' (12), plaintexts 3-6.
"""
from __future__ import annotations

import json
import random
import string
from pathlib import Path
from typing import Iterator

ALPHABET = string.ascii_lowercase
ALPHABET_12 = "abcdefghijkl"


def alphabet_for_key(key: str) -> str:
    """Infer the alphabet of an episode from its key length."""
    return string.ascii_lowercase[: len(key)]


def invert_key(key: str, alphabet: str | None = None) -> str:
    """Return the inverse permutation of a key string.

    If `key` is an encoding map (key[i] = ciphertext for plaintext alphabet[i]),
    the result is the decoding map (out[j] = plaintext for ciphertext alphabet[j]).
    Calling invert_key twice returns the original.
    """
    alphabet = alphabet or alphabet_for_key(key)
    out = [""] * len(alphabet)
    pos = {c: i for i, c in enumerate(alphabet)}
    for i, c in enumerate(key):
        out[pos[c]] = alphabet[i]
    return "".join(out)


def generate_episode(
    rng: random.Random,
    ep_id: int,
    alphabet: str = ALPHABET,
    min_len: int = 5,
    max_len: int = 15,
) -> dict:
    perm = list(alphabet)
    rng.shuffle(perm)
    key = "".join(perm)
    enc = {c: perm[i] for i, c in enumerate(alphabet)}

    length = rng.randint(min_len, max_len)
    plaintext = "".join(rng.choices(alphabet, k=length))
    ciphertext = "".join(enc[c] for c in plaintext)
    return {
        "id": ep_id,
        "plaintext": plaintext,
        "ciphertext": ciphertext,
        "key": key,
    }


def generate_dataset(
    seed: int,
    n: int,
    alphabet: str = ALPHABET,
    min_len: int = 5,
    max_len: int = 15,
) -> list[dict]:
    rng = random.Random(seed)
    return [generate_episode(rng, i, alphabet, min_len, max_len) for i in range(n)]


def write_jsonl(path: Path, episodes: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def decode_with_key(ciphertext: str, key: str) -> str:
    """Reference decoder. inv[key[i]] = ALPHABET[i]."""
    inv = {key[i]: ALPHABET[i] for i in range(len(ALPHABET))}
    return "".join(inv.get(c, c) for c in ciphertext)


def levenshtein(a: str, b: str) -> int:
    """Standard Levenshtein edit distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(
                cur[j - 1] + 1,         # insertion
                prev[j] + 1,            # deletion
                prev[j - 1] + cost,     # substitution
            )
        prev = cur
    return prev[-1]


def char_accuracy(pred: str, target: str) -> float:
    """1 - levenshtein / max(len). Bounded to [0, 1].

    Empty-vs-empty returns 1.0. Anything-vs-empty returns 0.0.
    """
    if not pred and not target:
        return 1.0
    denom = max(len(pred), len(target), 1)
    d = levenshtein(pred, target)
    return max(0.0, 1.0 - d / denom)


def exact_match(pred: str, target: str) -> bool:
    return pred == target


# ---------------------------------------------------------------------------
# Answer-tag parsing (shared between solo sanity check and dialog baseline)
# ---------------------------------------------------------------------------

import re as _re

ANSWER_RE = _re.compile(r"<answer>(.*?)</answer>", _re.IGNORECASE | _re.DOTALL)
_NON_LETTER_RE = _re.compile(r"[^a-z]")


def parse_answer(text: str) -> tuple[str | None, bool]:
    """Extract the LAST <answer>...</answer> match. Returns (plaintext, found).

    Strips any non-letter content inside the tag (whitespace, asterisks,
    quotes, punctuation), since instruction-tuned models often add cosmetic
    formatting even when asked not to.
    """
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None, False
    cleaned = _NON_LETTER_RE.sub("", matches[-1].lower())
    return cleaned, True


def tokens_before_answer(tok, raw_output: str) -> int | None:
    """Approximate token count up to (and not including) the first <answer> tag."""
    idx = raw_output.find("<answer>")
    if idx < 0:
        return None
    prefix = raw_output[:idx]
    return len(tok(prefix, add_special_tokens=False).input_ids)
