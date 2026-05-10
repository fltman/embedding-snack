# Results: phases 1–3

This directory holds the canonical result files for the work that has been completed (phases 1–3 of a planned 7-phase experiment). It is the place a skeptical reader should start.

What's here:

- `phase_2_solo_capacity_ceiling.jsonl` — 20 held-out episodes, single Qwen3-4B with the full task in one prompt. **The capacity ceiling.** What the model can do when given everything at once.
- `phase_2_dialog_text_baseline.jsonl` — 100 held-out episodes, two-instance dialog (3 round-trip exchanges) with the cipher key as an alphabet-aligned table in B's first user message. **The text baseline.** This is what a trained adapter has to beat.
- `phase_3_random_adapter_baseline.jsonl` — 100 held-out episodes, single-shot adapter pipeline with an untrained random-orthogonal adapter. **The chance floor.** Confirms the protocol learns nothing from an untrained projection.

All three were run on the same dataset (`data/cipher_v3_test.jsonl`, first 100 / first 20) with the same model (`Qwen/Qwen3-4B`, bf16, MPS, single shared instance), the same `<answer>...</answer>` parsing protocol, and the same scoring functions (Levenshtein-based char accuracy, exact match, format compliance, token usage).

## Headline numbers

| Condition | n | char_acc | EM | format compliance | mean tokens |
|---|---|---|---|---|---|
| Solo capacity ceiling | 20 | **0.903** | 0.75 | 1.00 | 75 |
| Two-model dialog (text) | 100 | **0.872** | 0.77 | 1.00 | 263 |
| Random adapter (chance floor) | 100 | **0.098** | 0.00 | 0.97 | 124 |

Random chance per character on the 12-letter alphabet ≈ 1/12 = 0.083. The random-adapter result (0.098, stdev 0.13) is statistically indistinguishable from chance.

## Accuracy distributions (over 100-episode runs)

**Dialog text baseline:**
- acc = 1.0 (perfect EM): **77 / 100**
- 0.5 ≤ acc < 1.0:        14 / 100
- 0 < acc < 0.5:           6 / 100
- acc = 0:                 3 / 100

**Random adapter:**
- acc = 1.0:               **0 / 100**
- 0.5 ≤ acc < 1.0:          2 / 100  (lucky chance hits)
- 0.25 ≤ acc < 0.5:         9 / 100
- 0 < acc < 0.25:          38 / 100
- acc = 0:                **51 / 100**

## How we got here (text baseline iteration history)

The text baseline was not clean on the first try. Seven configurations were measured before the dialog version reached the ceiling. Each is a tagged commit; `git log --tags` lists them.

| Tag | Config | char_acc | EM |
|---|---|---|---|
| `phase-2-1p7b-inadequate` | 1.7B, 26-letter alphabet, solo + two-shot | 0.028 / 0.080 | 0/20 |
| `phase-2-4b-marginal-26char` | 4B, 26-letter alphabet, solo with answer-tag protocol | 0.224 | 0/20 |
| `phase-2-4b-encmap-57pct` | 4B, 12-letter alphabet, solo, key as `a→X` (encoding map) | 0.570 | 0.25 |
| `phase-2-4b-decmap-90pct` | 4B, 12-letter, solo, key as `X→a` (decoding map, no inversion) | **0.903** | 0.75 |
| `phase-2-dialog-naive-17pct` | 4B, dialog, key as flat string in user | 0.172 | 0.01 |
| `phase-2-solo-system-key-78pct` | 4B, solo, key as table moved into system prompt (diagnostic) | 0.784 | 0.50 |
| `phase-2-dialog-tabkey-87pct` | 4B, dialog, key as alphabet-aligned table in user | **0.872** | 0.77 |

Key findings from the iteration:

1. **Qwen3-1.7B is genuinely incapable of 26-letter substitution lookup.** Solo accuracy 0.028 — below random chance per character (1/26 ≈ 0.038). Not a prompting problem.
2. **Qwen3-4B is marginal at 26 letters but solid at 12.** We reduced the task explicitly to give the model headroom. Documented honestly; the qualitative conclusions should generalize.
3. **Encoding-direction key (`a → X`) costs ~33pp.** The model spends its tokens inverting the permutation. Decoding-direction key (`X → a`) drops mean tokens from 294 → 75 and lifts accuracy from 0.57 → 0.90. Direct evidence of inversion overhead in text protocols — relevant for any length-pressure analysis later.
4. **Naïve dialog has one dominant failure mode** ("FM1": positional-index lookup, where B treats `key[ciphertext_position]` as the decode instead of `key[alphabet_position]`). Replacing the flat-string key (`"key is iachbgfljkde"`) with the alphabet-aligned table closed 70 of the 73-pp dialog gap by itself. Pure dialog-protocol overhead is only ~3pp.
5. **Key placement matters separately from format.** Moving the table from user-prompt to system-prompt costs ~12pp through reduced salience (model jumps to `<answer>` without consulting the table). Both effects measured separately in `phase-2-solo-system-key-78pct`.

The full failure-mode taxonomy with transcripts is in `../run_006_phase2_v3_dialog/dialog_failure_modes.md`. Sample episodes for each mode are saved under `../behavior_examples/`.

## Disclosures (relevant when interpreting these numbers)

- **All measurements use the same model (Qwen3-4B) on both Player A and Player B sides.** A=B is fine for measuring a text baseline, but the cross-model adapter claim that motivates the experiment requires different models. Phase 4 plans to swap B to `google/gemma-4-E4B-it`. Until that's done, "emergent cross-model communication" is the project's *intent*, not its measured result.
- **The 12-letter alphabet / 3–6 char plaintext task is a deliberate reduction from the spec's 26 / 5–15.** We measured both, found 4B was marginal at 26 letters, reduced explicitly so the model could perform the lookup reliably. Decision is documented in run-level notes.
- **The model still gets ~5/100 episodes wrong because of an unrelated training-data-prior leak** — B occasionally drops out of cipher-task context and produces plausible English (`'hello'`, `'hedge'`, `'lawl'`, `'unknown'`) instead of the cipher decode. Saved as `../behavior_examples/v3_dialog_english_word_hallucinations/`. Not addressable through prompt design alone; worth flagging as a noise floor.
- **Phase 3's adapter is single-shot** (one A→B vector per episode), not the spec's N=3 multi-turn. Multi-turn is a Phase 4 design decision still open.
- **`format_compliance` alone is not a useful signal of adapter success.** B emits `<answer>...</answer>` 97% of the time even when the signal carries zero information (random adapter). Char accuracy is the load-bearing metric.

## What's missing

Phases 4–7 of the original plan are not in this repo:

- Phase 4: train the adapter for 3000 episodes
- Phase 5: full evaluation battery (cross-model sanity, decoder probe, third-model interpretability, adapter ablations)
- Phase 6: length-pressure experiments (vector norm + turn count regularization, manifold-distance tracking)
- Phase 7: visualizations and writeup

The next concrete step is GPU compute (16 GB Mac is saturated; one Qwen3-4B instance already swaps), cross-model swap on B's side, and the training loop in `src/train.py` (not yet written).
