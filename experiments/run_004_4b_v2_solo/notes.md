# run_004_4b_v2_solo — Qwen3-4B, solo with `<answer>` tag, 12-letter alphabet

**Date:** 2026-05-10
**Model:** `Qwen/Qwen3-4B`, bf16, MPS, single instance.
**Dataset:** `data/cipher_v2_test.jsonl` — alphabet `a–l` (12), plaintext 3–6 chars.
**Prompt:** solo with `<answer>...</answer>` tag, `max_new_tokens=500`, key shown as encoding map (`a -> X`).
**N:** first 20 episodes of test split.

## Numbers

| Metric | Value | Gate |
|---|---|---|
| mean char_acc | 0.570 | ≥ 0.75 — **MISS** |
| format_compliance | 1.00 | ≥ 0.90 — **PASS** |
| exact_match_rate | 0.25 (5/20) | — |
| mean_tokens_used | 294 | — |
| mean_tokens_before_answer_tag | 284.5 | — |

Compared to run_003 (26-letter, 0.224 acc, 0.75 compliance, 0/20 EM) the 12-letter reduction lifted mean accuracy by ~35 percentage points and produced the first non-zero EM rate. Format compliance went from 0.75 to perfect after relaxing the regex (whitespace inside the tag) and bumping `max_new_tokens` to 500.

## Failure-mode taxonomy (3 buckets)

| Bucket | Count | Definition | Example |
|---|---|---|---|
| Perfect | 5/20 | EM=1, char_acc=1.0 | ep 2 `'khheh'`, ep 5 `'kicf'`, ep 9 `'jhfhf'`, ep 12 `'kki'`, ep 20 `'ihdb'` |
| Near-miss | 12/20 | char_acc 0.17–0.80, EM=0 | ep 18 `'dhhi'`→`'dhhii'` (0.80), ep 16 `'ljfd'`→`'ljcd'` (0.75), ep 3 `'gdfh'`→`'bdfh'` (0.75) |
| Total fail | 3/20 | char_acc 0.00 | ep 7 `'ekg'`→`'ade'`, ep 11 `'debfhg'`→`'bflika'`, ep 19 `'kkhhgl'`→`'bbllek'` |

The near-miss bucket dominates. Most errors are 1–2 wrong substitutions in an otherwise correct decode, suggesting the model can apply the cipher but loses fidelity on individual characters when the key is large enough to require sustained attention. The perfect bucket rules out "model fundamentally can't do this".

The total-fail bucket is more interesting: pattern matches the inverted-key-direction error first observed in run_002 ep 19 (the saved behavior example). Saved as `experiments/behavior_examples/v2_decoding_direction_failures/` for cross-reference. This is what motivates the Phase 2 (c) iteration in run_005: present the key as a decoding map (`X -> a`) and see if the total-fail bucket disappears.

## Memory-pressure observation (hard data, save for writeup)

During this 20-episode solo run on a 16 GB unified-memory Mac:
- `top` showed PhysMem 14 GB used, 918 MB unused, 854 MB compressor.
- `vm_stat` reported 27,552,819 swapins / 30,165,292 swapouts.
- The process spent ~3 minutes between "weights loaded" and the first generation, almost certainly because of memory thrashing as MPS pulled the model into wired memory.

This is with **a single 4B instance** in memory, no adapter, no second model. Phase 4 will require:
- Two model instances (A and B, when we move past the share-weights workaround), or
- Adapter optimizer state (~hundreds of MB at fp32) plus activations for gradient flow through B.

Either of those pushes us past 16 GB. Local hardware is the binding constraint for Phase 4, not iteration speed. The "we'll move to cloud later" plan from earlier is now confirmed by this measurement, not assumed. Plan a RunPod (or similar) provisioning step before Phase 4 starts.

## Decision

Run (c) — `cipher_v3_alphabet12_decmap` — solo on 4B. Then run full Phase 2 dialog baseline regardless of v3 outcome. Stop tweaking baseline after that.

Tagged: `phase-2-4b-encmap-57pct`.
