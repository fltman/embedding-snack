# run_006_phase2_v3_dialog — Qwen3-4B, two-instance dialog (naïve), v3 decmap

**Date:** 2026-05-10
**Model:** `Qwen/Qwen3-4B`, bf16, MPS, single instance shared between A and B (16 GB Mac).
**Dataset:** `data/cipher_v3_test.jsonl` — alphabet a–l, plaintext 3–6 chars, decoding-direction key.
**Protocol:** 3 round-trip exchanges (A→B, B→A, A→B, B→A, A→B, B→A), then explicit final prompt to B with `<answer>` tag.
**N:** 100 episodes.

## Numbers

| Metric | Solo (run_005) | **Dialog (this run)** | Δ |
|---|---|---|---|
| mean char_acc | 0.903 | **0.172** | **−0.73** |
| exact_match_rate | 0.75 | **0.01** (1/100) | −0.74 |
| format_compliance | 1.00 | 1.00 | — |
| mean_tokens_used | 75 | **293** | +218 (~4× bloat) |

**Distribution of accuracy:**
- acc = 0:        46 / 100
- 0 < acc < 0.5:  45 / 100
- 0.5 ≤ acc < 1:   8 / 100
- acc = 1.0:       1 / 100  (only episode 26: pt=`'igf'`)

72-percentage-point drop from solo to naïve dialog with the same model, same task, same data, same decoding-direction prompt. Format compliance is perfect (the `<answer>` protocol works in dialog). The drop is in algorithmic correctness, not formatting.

## Failure modes (see `dialog_failure_modes.md` for transcripts)

The dominant failure is **not** role confusion. B knows it's supposed to decode. A knows it's supposed to provide ciphertext and not produce the plaintext itself. The failure is structural:

1. **Positional-index error.** When B receives the key as a flat 12-char string in dialog (`"The cipher key is: iachbgfljkde"`) rather than as the table that solo gets (`a -> i / b -> a / c -> c / ...`), B can't reliably bind the lookup convention. The system-prompt text "the i-th character is the plaintext for the i-th ciphertext letter" is semantically ambiguous — does "i-th ciphertext letter" mean "letter at ciphertext position i" or "i-th letter of the alphabet"? B consistently chose the wrong reading: **map each ciphertext letter to `key[its position in the ciphertext]`** rather than `key[its alphabet position]`.

2. **Cascade lock-in.** Once B produces an answer, A relays it as truth ("the plaintext is X"). B reaffirms in the next round. Rounds 2 and 3 add no signal — they're confirmation echoes. No verification step.

3. **Token bloat.** Mean tokens per episode quadrupled (75 → 293). Most additional tokens are meta-talk: "the plaintext is X", "no further action is required", "Player B will produce it". Actual decoding work is a fraction.

4. **A never verifies.** A has the ciphertext and frequently sees the key (B states it). A could verify against ciphertext. Never does.

## Structural observations (separate from outcome)

This is not just "model is bad at dialog". It's that the dialog protocol I implemented forces specific failure-creating conditions:

- B's first turn has **two consecutive `user` messages** before B speaks: the initial "Talk to A" prompt and A's first message. Qwen3 is not trained on this structure.
- B receives the key as a flat string, not as a table. Solo provided the table. The convention isn't transferable to dialog without re-presenting it.
- A's instruction "do not output the final plaintext yourself" is followed in the wrong way — A doesn't produce the plaintext as final answer, but A does relay B's wrong answer back to B and lock it in.

These are dialog-protocol design issues, **not task or model issues**. The 90% solo result on the same data confirms the model can do the lookup; the 17% dialog result confirms our current protocol gets in its way.

Pending discussion with anders before redesigning. Not iterating yet.

Tagged: `phase-2-dialog-naive-17pct`.
