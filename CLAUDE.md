# Embedding Snack: Cross-Model Communication Without Tokens

A research experiment where two frozen LLMs talk to each other via hidden-state vectors instead of text, with a learned adapter as the only trainable bridge. We measure whether they can solve cooperative tasks more efficiently than with text, and we watch what happens to the "language" they invent when length pressure is applied.

This file is the spec. Read it end-to-end before writing code. Then propose the phase 1 plan and wait for approval before implementing.

---

## Why this exists

Most multi-agent LLM work routes communication through text. Tokenization is a bottleneck: it's lossy, slow, and forces the model to compress its full hidden state into a discrete vocabulary it shares with humans.

Two unanswered questions:

1. If we let two LLMs talk in their native representation (hidden states), can they solve tasks more efficiently?
2. If we apply selection pressure for shorter protocols, do their messages drift away from the manifold of human language? When we decode them back to text, is it still English?

This isn't novel in spirit — there's a real lineage (Lewis-style signaling games, the 2017 Facebook negotiation paper, the "emergent communication" subfield in multi-agent RL). What's missing is a clean cross-model implementation with modern instruction-tuned LLMs and an honest evaluation.

We're filling that gap. Proof of concept first, write-up after.

---

## Architecture

```
Model A (frozen)              Adapter (trainable)         Model B (frozen)
───────────────              ────────────────────         ───────────────
text in  ──────►  hidden_A  ──►  Linear + LN  ──►  hidden_B'  ──────►  text out
                  [d_A]          d_A → d_B          [d_B]
```

- **Model A**: Qwen3 8B Instruct (start with this — same family as Anders' Transcriber stack so we have known-good loading)
- **Model B**: Mistral Small 3 (different family for cross-tokenizer test). If GPU is tight, fall back to Qwen3 4B.
- **Adapter**: a single linear projection `d_A → d_B` plus LayerNorm. Start symmetric (same matrix used both directions). Asymmetric is a stretch goal.
- Both models stay frozen. Only the adapter has gradients.

### How we actually wire it up in HuggingFace

For Model A: run the forward pass, grab the last hidden state of the last layer at the final token position. `outputs.hidden_states[-1][:, -1, :]` after running with `output_hidden_states=True`.

For Model B: pass the adapted vector via `inputs_embeds=` instead of `input_ids=`. We prepend a short text prefix (`"<|user|>You received a signal:\n"`) embedded normally, then concatenate our adapted vector as a "fake token" embedding, then a closing prefix (`"\n<|assistant|>\n"`). Model B generates from there.

Gradient flows through Model B's frozen weights back to the adapter — this is standard prompt tuning math and works fine. Wrap Model A's forward in `torch.no_grad()` since we don't need gradients on its side.

---

## The task

Pick one task and stick with it for the whole experiment. Don't get clever with multi-task generalization yet.

**Recommended starting task: cooperative cipher decode.**

- Vocabulary: 26 lowercase letters
- Each episode generates a random monoalphabetic substitution cipher and a 5-15 character plaintext
- Model A sees the ciphertext only
- Model B sees the cipher key only
- They get N exchanges (start with N=3) of either text (baseline) or hidden states (treatment)
- Final action: Model B emits the decoded plaintext as text
- Fitness: character-level accuracy (Levenshtein-based)

This task is good because it's:
- Trivially solvable with text (lower bound: ~100% for both modes if the protocol works)
- Compressible (the optimal text protocol is just "send me the key" → "here it is" — so length pressure has somewhere to go)
- Easy to score automatically
- Easy to make harder later (longer plaintexts, polyalphabetic ciphers, multiple keys split between agents)

If this turns out too easy and converges in 50 steps, switch to a 20-questions game where B holds a secret object from a 1000-item taxonomy and A has to identify it within K turns.

---

## Training

Frozen backbones, trainable adapter. Standard supervised setup.

- **Loss**: cross-entropy on Model B's final output tokens against the ground-truth decoded plaintext
- **Optimizer**: AdamW, lr 1e-4, weight decay 0.01
- **Schedule**: cosine with 100-step warmup
- **Batch size**: 8 episodes (gradient accumulation if VRAM is tight)
- **Episodes**: aim for 3000-5000 — should converge well before that on the cipher task
- **Validation**: 200 held-out episodes every 100 steps

Log every episode to JSONL: episode_id, ciphertext, plaintext, key, all hidden vectors exchanged, B's raw output, character accuracy, loss.

---

## Phasing

Don't try to build everything at once. Each phase ends with something runnable and observable.

### Phase 1 — Setup & sanity check
- Load both models in bf16. Verify VRAM is OK on the available hardware.
- Implement the hidden-state extraction from Model A.
- Implement the `inputs_embeds` injection into Model B with the prefix/suffix wrapping.
- **Identity adapter test**: encode a simple sentence with Model A, set adapter = a fixed projection that maps A's space to B's space using a random orthogonal matrix (no training), feed into B, decode. Output will be garbage — that's fine. We're confirming the plumbing works without errors and that gradients flow when we enable them.
- Deliverable: a notebook showing one round-trip end-to-end with a debug print of shapes at each step.

### Phase 2 — Text baseline
- Implement the cipher-decode task generator.
- Run the task with normal text turns between A and B (no adapter). Use chat templates properly.
- Measure: accuracy, average turns to solution, total tokens exchanged.
- Deliverable: a baseline accuracy number and a markdown table. This is the bar we beat (or fail to beat).

### Phase 3 — Random adapter baseline
- Run the task with an untrained random adapter.
- Confirm accuracy is at chance level. Confirm decoded output is nonsense.
- Deliverable: confirmation that the protocol learns something nontrivial only when trained.

### Phase 4 — Train the adapter
- Build the training loop.
- Train for 3000 episodes, validate every 100.
- Save checkpoints every 500 steps.
- Deliverable: training curve, final validation accuracy. Compare against Phase 2.

### Phase 5 — Evaluation battery
Run all of these on a fixed 500-episode test set with the trained adapter:

1. **Cross-model sanity**: how does accuracy compare to the text baseline?
2. **Decoder probe**: take a sample of A's hidden states and run them through B's lm_head directly to get token distributions. Top-5 tokens at each position — is it English? Is it coherent? Save 50 examples for the writeup.
3. **Third-model interpretability**: load a third model (e.g., Llama 3.1 8B). Try to use it as Model B with the same adapter. If it works, the protocol is general. If not, it's specific to B's exact weights — also interesting, just a different finding.
4. **Adapter ablations**: replace the trained adapter with (a) identity, (b) random matrix, (c) the trained matrix with weights shuffled. Confirm the trained one wins clearly.

Deliverable: a results table covering all four.

### Phase 6 — Length pressure
- Modify the loss: `loss = ce_loss + λ · (vector_norm + turn_count)`
- Sweep λ over {0, 0.01, 0.1, 1.0}
- Train each setting for 1500 more episodes from the Phase 4 checkpoint
- Measure: does accuracy hold? Do messages get shorter / more compressed?
- Most importantly: track the **distance from the natural-text manifold**. Compute it like this: collect 10k embeddings of natural English sentences (from any random corpus) using Model B's embedding layer. Build a kNN index. For each adapter output during evaluation, measure mean distance to the k=5 nearest natural embeddings. Plot this distance over training.
- Deliverable: a 2x2 plot of (accuracy, manifold distance) vs λ.

### Phase 7 — Visualization & writeup
- UMAP of all adapter outputs across training. Color by epoch. Overlay natural-text embeddings as a baseline cloud.
- The drift plot from Phase 6.
- Decoded examples table: 10 rows of (ciphertext, what A "said" decoded back to text via B's lm_head, B's final output, ground truth).
- Generate a single hero figure that summarizes the finding. This is the figure that goes on LinkedIn.

---

## Tech stack

- Python 3.11+
- PyTorch 2.x
- transformers (≥ 4.45)
- accelerate
- bitsandbytes (only if quantization needed)
- umap-learn, scikit-learn
- matplotlib (figures should be readable in B&W and not look like default matplotlib — pick a clean style)
- pandas for results tables
- tqdm for loops

No wandb. Log everything to local JSONL. Anders runs experiments locally and likes data on disk.

---

## Repo layout

```
embedding-snack/
├── README.md                    # public-facing, written last
├── CLAUDE.md                    # this file
├── pyproject.toml
├── src/
│   ├── models.py                # loaders, hidden-state extraction, embeds injection
│   ├── adapter.py               # the trainable bridge
│   ├── tasks/
│   │   └── cipher_decode.py
│   ├── train.py
│   ├── eval.py
│   └── viz.py
├── experiments/
│   └── run_<timestamp>/
│       ├── config.yaml
│       ├── checkpoints/
│       ├── logs/
│       │   ├── training.jsonl
│       │   └── episodes/
│       ├── vectors/             # numpy dumps for offline viz
│       └── figures/
├── notebooks/
│   ├── 01_sanity.ipynb          # phase 1
│   ├── 02_text_baseline.ipynb   # phase 2
│   └── 99_writeup.ipynb         # phase 7
└── tests/
    └── test_pipeline.py         # quick smoke tests
```

---

## Hardware notes

Anders has a Mac. If we're running on Apple Silicon, switch to MLX or use the `mps` backend with bf16. Mistral Small 3 will be tight on a 64GB Mac — fall back to Qwen3 4B as Model B if needed. Both models in bf16 should fit on a 48GB CUDA card comfortably.

If running on a smaller machine for prototyping, drop to Qwen3 1.5B for both A and B during Phase 1-2 development. The pipeline should be model-size agnostic.

---

## Stretch goals (only if Phase 1-7 wraps cleanly)

- **Asymmetric adapter**: separate matrices A→B and B→A. Test if it changes anything.
- **Bidirectional**: let B reply with hidden states too, not just final text. Now we have a real protocol over multiple rounds.
- **Three-agent emergent communication**: add Model C and let them all share the adapter (or train pairwise adapters). Watch for protocol divergence vs convergence.
- **Discoverability**: train a small "translator" network that maps adapter outputs back to natural English. If it succeeds easily, the protocol is just compressed English. If it struggles, something genuinely new is happening.

---

## What "done" looks like

A GitHub repo where someone can clone, run `uv sync && python -m src.train --config configs/cipher_default.yaml`, and reproduce the headline result within a few hours of GPU time. Plus a writeup-ready set of figures and a results table.

The deliverable Anders cares about is the LinkedIn post and the article. The deliverable I care about is that the code is clean enough that the writeup is honest about what we measured and didn't measure.

---

## Working agreement

- Read this whole file before writing any code.
- After reading, propose the Phase 1 implementation plan in chat. Wait for approval.
- After each phase finishes, summarize what you found, what surprised you, and what should change before the next phase. Don't just barrel through.
- If a result looks too good or too clean, suspect the wiring before celebrating. The most common failure mode here is accidentally letting text leak through the channel — e.g., the adapter learning to encode the input tokens directly. Phase 5's decoder probe is specifically designed to catch this.
- Commit per phase with a tag.
- If you're stuck or uncertain about a research call (length-pressure schedule, whether to switch task, whether to bail on Mistral and go same-family), ask. Don't guess.
