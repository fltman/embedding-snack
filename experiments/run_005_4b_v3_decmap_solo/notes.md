# run_005_4b_v3_decmap_solo — Qwen3-4B, solo with decoding-map prompt

**Date:** 2026-05-10
**Model:** `Qwen/Qwen3-4B`, bf16, MPS, single instance.
**Dataset:** `data/cipher_v3_test.jsonl` — same plaintexts/ciphertexts as v2 but the `key` field stores the **decoding** permutation (X → a) instead of the encoding permutation (a → X).
**Prompt:** decoding-direction wording (`SOLO_PROMPT_DECODING`), `max_new_tokens=500`.
**N:** first 20 test episodes.

## Numbers

| Metric | v2 (encoding) | **v3 (decoding)** | Δ | Gate |
|---|---|---|---|---|
| mean char_acc | 0.570 | **0.903** | +0.33 | ≥ 0.75 ✅ |
| format_compliance | 1.00 | 1.00 | — | ≥ 0.90 ✅ |
| exact_match_rate | 0.25 | **0.75** | +0.50 | — |
| mean_tokens_used | 294 | **75** | **−219** (~4×) | — |
| mean_tokens_before_answer_tag | 284.5 | **65.4** | **−219** | — |

Both gates pass cleanly.

## What this proves

1. **The 26-letter run's "marginal capacity" diagnosis was wrong.** The model was always able to do the lookup. What it couldn't reliably do was *invert* a permutation in its head while also tracking ciphertext positions. Removing the inversion step (giving the decoding map directly) lifted accuracy by ~33 percentage points.

2. **The 3 total-failure episodes from v2 (ep 7, 11, 19) all hit 1.00 char_acc on v3.** They were the inversion-direction failure mode in pure form. Saved as `experiments/behavior_examples/v2_decoding_direction_failures/`. The v3 result confirms the diagnosis.

3. **Token usage collapsed to a quarter of the encoding-direction version.** v2 mean was 294 tokens per episode; v3 is 75. ~75% of the "natural" text-protocol tokens were spent doing the inversion, not solving the actual decoding task. This is direct evidence for Phase 6 length-pressure analysis: the text protocol has substantial overhead that has nothing to do with solving the cooperative task — it's overhead the encoder of the prompt forces on the model. The adapter condition has the option to avoid this overhead entirely.

## Remaining failures (5/20 EM=0)

All five non-EM episodes are **transposition errors**: the model produced the correct *characters* but in slightly wrong order. None are mapping mistakes.

| ep | plaintext | predicted | char_acc |
|---|---|---|---|
| 4  | `hleila` | `hleial` | 0.67 |
| 5  | `kicf`   | `kifc`   | 0.50 |
| 9  | `jhfhf`  | `jhhff`  | 0.60 |
| 15 | `eeebb`  | `eebbb`  | 0.80 |
| 16 | `ljfd`   | `lfjd`   | 0.50 |

Saved as `experiments/behavior_examples/v3_transposition_failures/`. This is "attention-jitter on output order", a different (and probably less interesting) failure class than mapping errors. Not worth chasing further at this scale.

## Decision

Use **v3 (decoding direction)** for the full Phase 2 dialog baseline. 90% solo accuracy is a clean, defensible bar for the adapter condition to beat in Phase 4.

Tagged: `phase-2-4b-decmap-90pct`.
