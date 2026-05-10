# run_010_random_adapter — Phase 3 random orthogonal adapter, single-shot

**Date:** 2026-05-10
**Model:** A=B=`Qwen/Qwen3-4B`, bf16, MPS, single shared instance.
**Adapter:** `Adapter(d_in=2560, d_out=2560)` = `Linear + LayerNorm`, weights initialized via `nn.init.orthogonal_` (CPU/fp32 → bf16 device copy), bias zero. Frozen, no gradient.
**Adapter init seed:** 42.
**Dataset:** `data/cipher_v3_test.jsonl`, 100 episodes.
**Pipeline:** A runs forward on `"The ciphertext is: {ct}"`, last-layer last-token hidden state extracted (`encode_a_signal`). Adapter projects. B receives system + user-init (decoding-table + framing) + a user message containing a sentinel position where the adapter's output is injected as a single fake-token embedding. B generates with `<answer>...</answer>` protocol. Single-shot — one A→B vector per episode.

## Numbers

| Metric | Value | vs Phase 2 dialog (run_009) | vs random chance |
|---|---|---|---|
| mean char_acc | **0.098** | 0.872 (−0.77) | ≈ 1/12 = 0.083 (within noise) |
| exact_match_rate | **0.000** (0/100) | 0.77 | — |
| format_compliance | 0.97 | 1.00 | — |
| mean tokens | 124 | 263 | — |
| signal vector norm | 50.6 (constant) | — | — |

Distribution of accuracy:
- acc = 0:       **51 / 100**
- 0 < acc < 0.25: 38 / 100
- 0.25 ≤ acc < 0.5: 9 / 100
- 0.5 ≤ acc < 1:  2 / 100  (lucky chance hits, not structure)
- acc = 1.0:     **0 / 100**

Predicted-output length 5.3 mean vs plaintext 4.6 — model produces strings of plausible plaintext length but with random characters.

## What this confirms

1. **Random adapter produces near-chance accuracy.** 0.098 vs 0.083 chance per character; difference within stdev (0.13) of the run. The protocol learns nothing from an untrained adapter.
2. **Format compliance is unaffected.** B still emits `<answer>...</answer>` 97% of the time even when the signal carries no information. So format compliance alone is not a useful signal of "the adapter works" — char_acc is the load-bearing metric.
3. **Token usage drops vs trained-text-protocol.** 124 vs 263 in dialog. The model spends fewer tokens hedging because it has nothing concrete to reason about; it just emits a quick guess.
4. **The 12-letter alphabet's chance floor is clearly visible.** No mass at high accuracy. The 2 episodes with acc ≥ 0.5 are random hits (running mean stays flat).
5. **English-word hallucination pattern persists** (e.g. ep 100 pred=`'unknown'`). Same training-data prior leakage seen in run_009 (FM5). This is a B-side phenomenon, not adapter-related.

## What Phase 4 must beat

Solo capacity: 0.903 char_acc, 0.75 EM (run_005).
**Dialog text baseline: 0.872 char_acc, 0.77 EM (run_009).**
Random adapter floor: 0.098 char_acc, 0.00 EM (this run).

Trained adapter must clear the dialog text baseline at minimum to demonstrate vector-communication is competitive with text protocols. Clearing solo would mean vector communication is *better* than the model can do with full text context — that's the headline claim.

## Implementation notes for Phase 4

- `src/baselines/adapter_pipeline.py` is the runnable pipeline. Single-shot. Multi-turn (N=3 hidden state exchanges) is a Phase 4 design decision still open.
- The adapter's input dim (2560) equals output dim because A=B=Qwen3-4B. After cross-model swap to Gemma 4-E4B-it on B's side (Phase 4), `d_b` will differ and the adapter becomes a non-square projection. Verify `model_b.config.hidden_size` after loading Gemma 4 — likely not 2560.
- `signal_norm` is currently constant (50.6) because random orthogonal × LayerNorm produces a fixed-norm vector regardless of input. After training this should track input information content. Add `signal_norm` to Phase 4 logging.
- Adapter weights need `requires_grad_(True)` for Phase 4. Currently set to `False` in `scripts/03_random_adapter_baseline.py` since Phase 3 is forward-only.

Tagged: `phase-3-complete`.
