# run_001 — Phase 2 text baseline, Qwen3-1.7B (inadequate)

**Date:** 2026-05-09
**Model:** `Qwen/Qwen3-1.7B` (A and B share weights — same instance, different system prompts)
**Dataset:** `data/cipher_test.jsonl`, first N episodes; seed=3 fixed
**Outcome:** Model is genuinely too weak for the task. Promoting to Qwen3-4B.

## Numbers

| Setup | n | Mean char-acc | Exact match | Mean tokens |
|---|---|---|---|---|
| Smoke (3-round dialog, no example) | 5 | 0.093 | 0/5 | ~720 |
| **Solo upper bound** (single prompt, full info via `a -> X` map) | **20** | **0.028** | **0/20** | ~13 |
| **Two-shot dialog** (worked example added to B's system prompt) | **20** | **0.080** | **0/20** | ~539 |

Random per-character chance for 26-letter alphabet ≈ 0.038. Solo accuracy is **below random**.

## Why this is conclusive

Solo upper bound rules out the "protocol overhead" hypothesis. When a single model sees the entire problem in one prompt — explicit `a -> X` mapping, ciphertext, instruction to decode — it still cannot do it.

Failure modes from `logs/transcripts_solo_1p7b.jsonl`:

- **Echo:** episode 0 returned the ciphertext verbatim as "plaintext"
- **Single-character outputs:** 8/20 episodes returned a 1-letter answer regardless of plaintext length (5–15 chars)
- **Random gibberish:** the rest were strings of plausible length but uncorrelated with the actual mapping

Two-shot dialog with a worked example also fails to recover (~8% mean acc, no exact matches), so it isn't a prompting fix.

## Calibration note for the writeup (do not lose this)

The original 5-episode smoke gave mean char-acc 0.093 in dialog mode. That number is **inflated by length-matching noise**: dialog runs produce ~10-character outputs (matching plaintext lengths), so chance per-character matches contribute meaningfully to a Levenshtein-based score. The solo run scores lower precisely because 1-character outputs have fewer chars to match by accident.

Implication: any cipher-decode mean char-accuracy below ~5% on this task at this scale is **indistinguishable from "model produces a string of plausible length, content meaningless"**. The real signal floor is well above per-char random chance. We should be explicit about this when reporting baseline numbers in the writeup, to avoid an honest reader concluding "the adapter only had to beat 9%".

## Decision

- Promote both A and B to `Qwen/Qwen3-4B`.
- Keep `share_model_weights=True` (single in-memory instance) for Phase 2 to fit on 16 GB unified-memory Mac.
- Cross-model swap (B = different family, e.g. Llama 3.2 1B) deferred to Phase 4, after we've confirmed 4B-on-both produces a measurable text baseline.
- Cloud (RunPod or similar) is the planned exit point for Phase 4 when we need two separate 4B instances for adapter-gradient flow.

## Reproducing

```
git checkout phase-2-1p7b-inadequate
uv run python scripts/sanity_checks.py
```

Raw transcripts: `logs/transcripts_solo_1p7b.jsonl` (20 episodes, solo prompt).
