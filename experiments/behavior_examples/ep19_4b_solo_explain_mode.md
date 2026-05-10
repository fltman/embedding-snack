# Behavior example: Qwen3-4B, solo cipher decode, episode 19

Captured before applying the `<answer>...</answer>` formatting fix in `scripts/sanity_checks.py`. After that fix the model is likely to suppress its work and emit only the final answer, so this kind of transcript becomes rare. Saved as a writeup-relevant artifact.

## Setup

- Model: `Qwen/Qwen3-4B`, bf16, MPS, single instance.
- Prompt: solo upper-bound prompt with `a -> X` mapping, `max_new_tokens=50`.
- Dataset: `data/cipher_test.jsonl`, episode id 19.
- Run: `experiments/run_002_4b_solo/`.

## Episode

```
ciphertext   : sldllsfyl
plaintext_gt : alrllagwl
key          : sojmeafiuwxlrtqvbdkphnycgz
```

Verify the ground truth: `key[0]='s'`, so plaintext `'a'` (alphabet position 0) encodes to ciphertext `'s'`. To decode `'s'`: find `'s'` in key at position 0 → plaintext `'a'`. ✓

## Raw model output

```
sldllsfyl

Using the substitution cipher key:

s -> k  
l -> l  
d -> m  
l -> l  
l -> l  
s -> k  
f -> a  
y -> g  
l -> l

So the
```

(Cut off at 50 tokens before reaching the consolidated plaintext.)

## What the parser captured

`parse_plaintext` (longest plausible `[a-z]+` run) extracted **`'substitution'`** from the prose `"Using the substitution cipher key"`. Char accuracy: 0.0. Exact match: 0.

## What the model actually claimed (implicit decode)

Reading the per-character mappings the model wrote out:

| ct | model says | gt position | gt char | model correct? |
|----|------------|-------------|---------|----------------|
| s  | k          | 0           | a       | ✗              |
| l  | l          | 1           | l       | ✓              |
| d  | m          | 2           | r       | ✗              |
| l  | l          | 3           | l       | ✓              |
| l  | l          | 4           | l       | ✓              |
| s  | k          | 5           | a       | ✗              |
| f  | a          | 6           | g       | ✗              |
| y  | g          | 7           | w       | ✗              |
| l  | l          | 8           | l       | ✓              |

Implicit predicted plaintext: `klmllkagl`. Per-character accuracy if the parser had captured this: **4/9 ≈ 0.44**, not 0.0.

## The interpretation bug

The model's `s -> k` mapping is consistent with reading the key in the *opposite* direction from our convention. Our key string is the encoding map: `key[i]` is the ciphertext image of alphabet letter `i` (so position 0 maps `'a'` → `'s'`). The model appears to read it as a decoding map: position 18 (`'s'` in alphabet) maps to `key[18] = 'k'`, treating `'s'` as the plaintext side.

This explains every wrong mapping the model produced. Within its (inverted) interpretation, the lookups are mechanically correct.

## Why this matters for the writeup

Three findings packed into one episode:

1. **The 0% char-accuracy reported by the metric is meaningless here.** The model produced a plausible character-level decode, but the parser keyed on prose vocabulary instead. This is what motivated the `<answer>...</answer>` tag protocol.

2. **The model can do the lookup mechanically — when it has the right direction.** Capacity isn't the issue. The "explain mode" prose format actually exposes that the model is performing a structured operation, just on a misinterpreted key.

3. **Text-mode reasoning overhead is real and visible.** The model spent ~50 tokens reciting the per-character decoding table before it would have arrived at the consolidated plaintext. In the adapter condition (Phase 4+) B has no equivalent scratch space — it must commit to plaintext directly from the vector. The compression ratio here (one vector vs ~50 tokens of prose) is exactly the kind of efficiency claim the project is designed to test.
