# run_009_dialog_tabkey — Qwen3-4B dialog with table-formatted key in user (α fix)

**Date:** 2026-05-10
**Model:** `Qwen/Qwen3-4B`, bf16, MPS, single instance shared.
**Dataset:** `data/cipher_v3_test.jsonl` (alphabet a-l, 3-6 char plaintext, decoding-direction key), first 100 episodes.
**Protocol:** N=3 round-trip exchanges, then explicit `<answer>...</answer>` final-emission prompt.

## Variables changed vs run_006 (`phase-2-dialog-naive-17pct`)

| Variable | run_006 | run_009 |
|---|---|---|
| B's first user message | `"The cipher key is: iachbgfljkde\n\nTalk to Player A."` | Block 1 from solo's `SOLO_PROMPT_DECODING` (with structurally required adaptation: "below" → "that Player A will provide"; ciphertext-line removed; answer-tag instruction removed since it's in FINAL_PROMPT) followed by the alphabet-aligned table, followed by `"Talk to Player A."` |
| Everything else (system prompts, n_exchanges, FINAL_PROMPT, A's prompts, answer-tag protocol, max_new_tokens, dataset, model, key_direction) | identical | identical |

## Numbers

| Metric | Solo (run_005) | Naive dialog (run_006) | **Tabkey dialog (run_009)** |
|---|---|---|---|
| mean char_acc | 0.903 | 0.172 | **0.872** |
| exact_match_rate | 0.75 | 0.01 | **0.77** |
| format_compliance | 1.00 | 1.00 | **1.00** |
| mean tokens / ep | 75 | 293 | **263** |

**Distribution of accuracy in run_009:**
- acc = 0:        7 / 100
- 0 < acc < 0.5: 10 / 100
- 0.5 ≤ acc < 1: 6 / 100
- acc = 1.0:    77 / 100  (= EM rate)

## What this proves

- **Almost the entire 73-percentage-point dialog drop in run_006 was attributable to FM1** (positional-index lookup error caused by ambiguous flat-string key presentation). Replacing the flat string with the alphabet-aligned table recovered 70 of those 73 points without changing anything else about the dialog protocol.
- **Pure dialog-protocol overhead is only ~3pp.** Solo char_acc 0.903 vs dialog char_acc 0.872. EM is actually *higher* in dialog (0.77 vs 0.75), within run-to-run noise.
- **Token bloat persists** but is no longer error-creating. Mean tokens 263 vs 75 in solo. The cascade lock-in (FM2) is still present — A relays B's answer, B confirms, etc. But because the answer is correct in 77% of cases, the cascade cements correct results rather than wrong ones. Most extra tokens are now formality, not failure.
- **(β) — turn-structure redesign — is not necessary for a clean baseline.** The remaining 3pp gap doesn't justify ablating the dialog protocol further.

## New failure mode emerged: English-word hallucination

In a small fraction of episodes (~5/100), B's output is a plausible English word that ignores the actual cipher table:
- ep 52: pt=`'ljljk'` → pred=`'hello'`
- ep 61: pt=`'lifl'` → pred=`'lawl'`
- ep 70: pt=`'lde'` → pred=`'fil'`
- ep 83: pt=`'edjeg'` → pred=`'hedge'`

Saved to `experiments/behavior_examples/v3_dialog_english_word_hallucinations/`. The model occasionally drops out of the cipher-task context and falls back to "produce a plausible English string of similar length". Worth a note in the writeup as an artifact of training-data prior leaking through. Not addressable through prompt design without further work.

## Decision

Phase 2 dialog baseline = **0.872 char_acc, 0.77 EM, 263 mean tokens**. Both gates pass with comfortable margin. This is the bar Phase 4's adapter condition has to beat to demonstrate vector-communication superiority.

Tagged: `phase-2-dialog-tabkey-87pct`.

## Cumulative phase-2 ledger

| Tag | Setup | char_acc | EM |
|---|---|---|---|
| `phase-1-passed` | Plumbing only | — | — |
| `phase-2-1p7b-inadequate` | 1.7B 26-letter, solo+twoshot | 0.028 / 0.080 | 0/20 |
| `phase-2-4b-marginal-26char` | 4B 26-letter, solo answer-tag | 0.224 | 0/20 |
| `phase-2-4b-encmap-57pct` | 4B 12-letter, solo encoding map | 0.570 | 0.25 |
| `phase-2-4b-decmap-90pct` | 4B 12-letter, solo decoding map | 0.903 | 0.75 |
| `phase-2-dialog-naive-17pct` | 4B dialog, key as flat string | 0.172 | 0.01 |
| `phase-2-solo-system-key-78pct` | Diagnostic: key in system | 0.784 | 0.50 |
| **`phase-2-dialog-tabkey-87pct`** | **4B dialog, key as table in user** | **0.872** | **0.77** |
