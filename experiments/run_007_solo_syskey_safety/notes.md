# run_007_solo_syskey_safety — diagnostic, NOT a baseline

**Date:** 2026-05-10
**Model:** `Qwen/Qwen3-4B`, bf16, MPS, single instance.
**Dataset:** `data/cipher_v3_test.jsonl`, first 20 episodes.
**Prompt mode:** `--mode system-key` — key as alphabet-aligned table in **system** prompt, ciphertext in user prompt.

This run is a controlled measurement of one variable: **moving the cipher key from the user prompt (run_005) to the system prompt** while keeping the table format identical.

## Numbers

| Metric | run_005 (key in USER) | **run_007 (key in SYSTEM)** | Δ |
|---|---|---|---|
| mean char_acc | 0.903 | **0.784** | −0.12 |
| exact_match_rate | 0.75 | **0.50** | −0.25 |
| format_compliance | 1.00 | 1.00 | — |
| mean_tokens_used | 75 | **26** | −49 |
| mean_tokens_before_answer_tag | 65.4 | **16.4** | −49 |

## What this measures

Moving the key from USER → SYSTEM costs ~12 percentage points of accuracy on this model and this task. The token-budget collapse (75 → 26) explains part of it: the model frequently jumps straight to `<answer>...</answer>` with `pre_tag=0` and produces gibberish without consulting the table. When key was in user position, the model spent ~65 tokens applying the lookup deliberately.

This is a salience effect: instructions in system prompts compete less for attention during generation than content in the immediately preceding user turn.

## Why this is NOT the dialog baseline

Originally I conflated two changes:
1. Reformat key from flat string to alphabet-aligned table (the actual disambiguation fix anders specified).
2. Move key from user position to system position (an unrelated structural change).

Run_007 measures only (2), with the table format already in place. It is informative as standalone data but it is not the apples-to-apples version of run_005 needed to validate the disambiguation fix.

The actual phase-2 dialog baseline (forthcoming, run_008) keeps key in user position (matching run_005 and run_006) and changes only the format.

Tagged: `phase-2-solo-system-key-78pct`.
