# run_003_4b_solo_tagged — Qwen3-4B, solo with `<answer>` tag (26-char alphabet, marginal)

**Date:** 2026-05-10
**Model:** `Qwen/Qwen3-4B`, bf16, MPS, single instance.
**Dataset:** `data/cipher_test.jsonl` (26-letter alphabet, 5–15 char plaintext), first 20 episodes.
**Prompt:** solo with explicit `<answer>...</answer>` tag protocol, `max_new_tokens=300`.

## Numbers

| Metric | Value | Anders' gate |
|---|---|---|
| mean char_acc | 0.224 | ≥ 0.50 — **MISS** |
| format_compliance | 0.75 | ≥ 0.80 — **MISS** |
| exact_match_rate | 0.00 | — |
| mean_tokens_used | 233 | — |
| mean_tokens_before_answer_tag | 208.5 | (compression target for adapter) |

## Three failure modes diagnosed

1. **Parser too strict.** Episode 0 emitted `<answer>haphyapmztc hp</answer>` (with internal whitespace). The strict regex `<answer>\s*([a-zA-Z]+)\s*</answer>` rejected it. Counted as compliance failure but the model *did* try to comply. Fix: relax the regex to capture anything between tags, then strip non-letters.
2. **300-token budget too small.** Episodes 5, 6, 7, 8 hit `tok=300, pre_tag=None` mid-decode, never reaching the closing tag. Fix: `max_new_tokens=500`.
3. **Genuine substitution errors.** Even after the above fixes, the model produces wrong character-level mappings on the 26-letter alphabet. Best episodes hit 0.71–0.90 char_acc; most land at 0.10–0.30. Capacity is on the edge; the model can perform the lookup *sometimes*.

## Best/worst episode examples

- **Best:** ep 16, pt=`'qqkhemusce'`, pred=`'qqkhemuce'`, acc=0.90 (one missing 'c').
- **Best 2:** ep 13, pt=`'ogmwkss'`, pred=`'ogmvks'`, acc=0.71.
- **Worst with tag:** ep 19, pt=`'gcoyiwsdwpysti'`, pred=`'bfamsczicmqmzrs'`, acc=0.00.
- **Tag missing despite finishing:** ep 0, internal whitespace in tag.
- **Tag missing due to timeout:** eps 5–8.

## Decision (anders, 2026-05-10)

Reduce task complexity rather than scale model up. Move to **alphabet a–l (12 letters)**, plaintext **3–6 characters**, full permutation key. Hypothesis: 26-letter lookup is right at 4B's edge; halving the alphabet should land it solidly in capable territory without changing the scientific question. Stay on local 16 GB Mac for Phases 1–3; cloud move deferred to Phase 4 as originally planned.

This run is the "why we reduced to 12 letters" reference. Tagged `phase-2-4b-marginal-26char`.

## Reproducing

```
git checkout phase-2-4b-marginal-26char
uv run python scripts/sanity_checks.py --solo-only --out-name run_003_4b_solo_tagged
```
