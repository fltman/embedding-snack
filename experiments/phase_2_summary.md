# Phase 2 summary

## What we measured

The cipher-decode task: each episode is a random monoalphabetic substitution cipher plus a random plaintext. Player A sees the ciphertext only; Player B sees the cipher key only; the goal is for B to emit the decoded plaintext. We measure character-level accuracy (Levenshtein-based) and exact-match rate over 100 held-out test episodes, plus format compliance (did B emit a parseable answer at all) and token usage (compression-ratio target for Phase 4).

Phase 2's job was to establish a clean text-only baseline that the adapter condition in Phase 4 has to beat. Two configurations were measured: **solo** (single model with full prompt — capacity ceiling) and **dialog** (two-instance chat over N=3 round-trips — apples-to-apples comparison with the adapter setup).

## What we learned

- **Qwen3-1.7B is genuinely incapable** of a 26-letter substitution cipher even with the entire problem in one prompt (mean acc 0.028, below random per-character chance). Conclusive in `phase-2-1p7b-inadequate`.
- **Qwen3-4B is marginal at 26 letters** (0.224 solo, 0/20 EM) but **solid at 12 letters / 3-6 char plaintexts** (0.903 solo with answer-tag protocol). Reducing alphabet size halves attention-spread without changing the scientific question. Documented as the "we tuned the task to make it work" disclosure point.
- **Encoding-map key (`a→X`) costs ~33pp** vs decoding-map key (`X→a`) on the same data. The inversion overhead is the model spending ~75% of its tokens generating per-character mappings. The decoding-map version drops mean tokens from 294 → 75. This is direct evidence of inversion overhead in text protocols — **Phase 6 length-pressure relevance**.
- **Flat-string key in dialog causes FM1 (positional-index lookup error)**, the dominant 70-pp failure mode in naïve dialog. The model interprets "i-th ciphertext letter" as ciphertext-position when no alphabet-aligned table is present.
- **Key placement matters separately from format.** Moving the table from user-prompt to system-prompt costs ~12pp through reduced salience (model jumps to `<answer>` without consulting). Both effects measured separately in `phase-2-solo-system-key-78pct`.

## Choices and why

- **Task config**: alphabet a-l (12 letters), plaintext 3-6 characters, full permutation. Reduced from a-z / 5-15 explicitly to land 4B in capable territory.
- **Key direction**: decoding map (`X → plaintext`). Eliminates inversion overhead.
- **Key presentation**: alphabet-aligned table (`a -> i / b -> a / ...`), placed in B's first user message (not system).
- **Model**: A = B = Qwen3-4B, single shared instance. Required by 16 GB Mac unified-memory limit.
- **Protocol**: 3 round-trip exchanges + explicit `<answer>...</answer>` final-emission turn. The protocol contributes only ~3pp overhead vs solo once FM1 is removed.

## Final baseline numbers (the bar Phase 4 has to beat)

| | Solo (run_005) | **Dialog (run_009)** |
|---|---|---|
| mean char_acc | 0.903 | **0.872** |
| exact_match_rate | 0.75 | **0.77** (77/100) |
| format_compliance | 1.00 | 1.00 |
| mean tokens / ep | 75 | **263** |

## Residual uncertainties

- **A == B intra-model**: the entire Phase 2 baseline is single-model. The cross-model claim that motivates the experiment is untested. Decision needed before Phase 4.
- **English-word hallucination** (~5% of dialog episodes): B occasionally drops out of cipher context and produces plausible English (`hello`, `hedge`, `lawl`). Training-data prior leaking. Not addressable through prompt design alone.
- **Transposition errors** in non-EM episodes: model produces correct characters in slightly wrong order. Attention-jitter, not algorithm error.
- **Memory ceiling on 16 GB Mac confirmed**: single 4B instance + activations already swaps heavily (27M swapouts during run_004). Phase 4 (two instances + adapter optimizer state) does not fit. Cloud move planned post-Phase 3.

## Behavior examples preserved

- `experiments/behavior_examples/ep19_4b_solo_explain_mode.md` — Qwen3 reasoning trace before answer-tag fix
- `experiments/behavior_examples/v2_decoding_direction_failures/` — inverted-key-direction errors on encoding map
- `experiments/behavior_examples/v3_transposition_failures/` — character-order errors at solo-90%
- `experiments/behavior_examples/dialog_failure_samples/` — the 9 representative naïve-dialog failures
- `experiments/behavior_examples/v3_dialog_english_word_hallucinations/` — training-data-prior leakage in tabkey dialog

## Next

Phase 3: random adapter baseline (forward-only, fits locally). Confirms that an untrained adapter produces near-zero accuracy and that the protocol learns something non-trivial only when trained. Then cloud move + Phase 4.
