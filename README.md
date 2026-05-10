# Embedding Snack

This is the unfinished half of an experiment looking for a continuation. See [post-länk].

Two frozen LLMs talk to each other through a learned adapter that bridges their hidden-state spaces, instead of through text. The full experiment is seven phases; this repo contains phases 1 through 3 — enough to establish a clean text-only baseline and confirm the protocol learns nothing without training. Phase 4 (training the adapter) requires GPU hardware that wasn't available locally.

If you want to take it from here, the place to start is [`experiments/results/RESULTS.md`](experiments/results/RESULTS.md).

## What's the experiment

Most multi-agent LLM work routes communication through text. Tokenization is a bottleneck: it's lossy, slow, and forces the model to compress its full hidden state into a discrete vocabulary it shares with humans. Two questions worth asking:

1. If two LLMs talk in their native representation (hidden states), can they solve cooperative tasks more efficiently than via text?
2. Under selection pressure for shorter protocols, does their "language" drift away from the manifold of human language?

This isn't novel in spirit — there's a real lineage (Lewis-style signaling games, the 2017 Facebook negotiation paper, the broader "emergent communication" subfield in multi-agent RL). What's missing is a clean cross-model implementation with modern instruction-tuned LLMs and an honest evaluation. That's the gap this is trying to fill.

The full plan is in [`CLAUDE.md`](CLAUDE.md).

## Status

| Phase | What | Status | Result |
|---|---|---|---|
| 1 | Plumbing & sanity check | ✅ done | One round-trip end-to-end works on MPS bf16; gradients flow through the adapter |
| 2 | Text baseline | ✅ done | 0.872 char_acc / 0.77 EM in two-model dialog with same model on both sides |
| 3 | Random adapter baseline | ✅ done | 0.098 char_acc / 0/100 EM — chance level confirmed; protocol learns nothing untrained |
| 4 | Train the adapter | ⛔ blocked on hardware | — |
| 5 | Evaluation battery | not started | — |
| 6 | Length-pressure experiments | not started | — |
| 7 | Visualization & writeup | not started | — |

Phase 4 needs two model instances in memory simultaneously plus optimizer state and activations for backprop. A single `Qwen3-4B` instance already saturates the 16 GB Mac this was developed on; running two instances + adapter gradients does not fit. The natural next step is RunPod or similar with a 24 GB+ GPU.

## What we actually measured

Cooperative cipher decode. Each episode generates a random monoalphabetic substitution cipher and a 3–6 character plaintext from a 12-letter alphabet. Player A sees only the ciphertext; Player B sees only the cipher key. The goal is for B to emit the decoded plaintext.

We measure character-level accuracy (Levenshtein-based), exact-match rate, format compliance (did B emit a parseable answer), and total tokens used (compression-ratio target for Phase 6).

Three baselines, all on the same 100 held-out test episodes:

| Condition | char_acc | EM | mean tokens |
|---|---|---|---|
| Solo capacity ceiling (single model, full prompt) | 0.903 | 0.75 | 75 |
| **Two-model dialog (text)** | **0.872** | **0.77** | 263 |
| Random adapter (chance floor) | 0.098 | 0.00 | 124 |

The trained adapter has to clear the dialog text baseline (0.87) to be interesting. Clearing the solo capacity ceiling (0.90) would be the headline claim.

Full numbers, the iteration history (we ran seven configurations on the text baseline before the dialog version was clean), and the failure-mode taxonomy are in [`experiments/results/RESULTS.md`](experiments/results/RESULTS.md).

## Important honest disclosures

- **Same model on both sides.** All measurements so far use `Qwen/Qwen3-4B` for both Player A and Player B. The cross-model claim that motivates the experiment (different models with different tokenizers and embedding geometries) is untested. The plan is to swap B to `google/gemma-4-E4B-it` for Phase 4 onward, but that hasn't happened yet. Until it does, "emergent cross-model communication" is the project's *intent*, not its measured result.
- **Task complexity was reduced from spec.** Original task was a 26-letter alphabet with 5–15 character plaintexts. `Qwen3-4B` is marginal at that scale (0.224 solo char_acc). We reduced to a 12-letter alphabet with 3–6 character plaintexts so the model could perform the lookup reliably. The decision and reasoning are documented in `experiments/run_001/notes.md` and `experiments/run_004_4b_v2_solo/notes.md`. The qualitative conclusions should generalize; the quantitative numbers are at this scale.
- **Several failure modes were prompt-design artifacts, not model limits.** Naïve dialog scored 0.172 char_acc on the same task where solo scored 0.903. Almost the entire 73-pp gap was a single failure mode (FM1: positional-index lookup error caused by ambiguous flat-string key presentation). Replacing the flat-string key with an alphabet-aligned table closed 70 of those 73 points without changing the task. Anyone continuing this work should expect prompt-design issues to dominate measurement noise until the dialog protocol is fully nailed down.

## Reproducing

Requires Python ≥ 3.11 and `uv`.

```bash
git clone https://github.com/fltman/embedding-snack
cd embedding-snack
uv sync

# Generate datasets (idempotent — skips if already present)
uv run python scripts/generate_cipher_data.py

# Phase 1 sanity: round-trip + gradient check
uv run python scripts/01_sanity.py

# Phase 2 solo baseline (20 episodes, ~2 min on cached MPS)
uv run python scripts/sanity_checks.py --solo-only \
    --dataset data/cipher_v3_test.jsonl --key-direction decoding \
    --out-name run_solo_repro

# Phase 2 dialog baseline (100 episodes, ~1 hour on MPS with swap)
uv run python scripts/02_text_baseline.py --n 100 \
    --dataset data/cipher_v3_test.jsonl --key-direction decoding \
    --out-name run_dialog_repro

# Phase 3 random adapter baseline (100 episodes, ~50 min on MPS)
uv run python scripts/03_random_adapter_baseline.py --n 100 \
    --dataset data/cipher_v3_test.jsonl --key-direction decoding \
    --out-name run_random_adapter_repro
```

All key results are tagged in git — `git log --tags --oneline` lists them. Browsing them in order tracks how the baseline got to its final shape.

## Repo layout

```
embedding-snack/
├── CLAUDE.md               # the original experiment spec
├── README.md               # this file
├── LICENSE                 # MIT
├── pyproject.toml          # deps via uv
├── src/
│   ├── models.py           # loaders, hidden-state extraction, inputs_embeds injection
│   ├── adapter.py          # the trainable bridge (Linear + LayerNorm)
│   ├── tasks/cipher_decode.py  # task generator, scorer, answer-tag parser
│   └── baselines/
│       ├── text_chat.py        # text-only dialog runner
│       └── adapter_pipeline.py # adapter-mediated single-shot pipeline
├── scripts/                # phase runners
├── data/cipher_*.jsonl     # versioned task datasets (committed)
├── notebooks/              # phase-1 sanity round-trip notebook
├── experiments/
│   ├── results/            # canonical results — start here
│   ├── run_001 .. run_010  # full run histories with notes.md per run
│   └── behavior_examples/  # transcripts illustrating specific failure modes
└── tests/                  # (empty — testing is via the run logs themselves)
```

## License

MIT. See [LICENSE](LICENSE).
