"""Phase 1 sanity check: A -> adapter -> B end-to-end, plus gradient flow.

Run:
    uv run python scripts/01_sanity.py

What this verifies:
    * Both models load on the detected device (mps / cuda / cpu).
    * encode_to_hidden returns the expected shape from Model A.
    * A random orthogonal adapter projects A's hidden state to B's d_model.
    * decode_from_hidden generates text without errors. Output IS garbage; that
      is correct for an untrained adapter. We are not measuring quality here.
    * Enabling grads on the adapter and backprop'ing a dummy loss produces a
      finite, nonzero gradient on the adapter weights. This proves the chain
      A.no_grad -> adapter (trainable) -> B.frozen -> loss is wired correctly.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable when run as a script from the repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.adapter import Adapter, init_orthogonal  # noqa: E402
from src.models import (  # noqa: E402
    DEFAULT_MODEL_A,
    DEFAULT_MODEL_B,
    decode_from_hidden,
    encode_to_hidden,
    load_frozen,
    pick_device,
)


def main() -> None:
    device = pick_device()
    print(f"[device] {device}")

    print("\n=== Step 1: Load Model A ===")
    model_a, tok_a = load_frozen(DEFAULT_MODEL_A)

    print("\n=== Step 2: Load Model B ===")
    if DEFAULT_MODEL_B == DEFAULT_MODEL_A:
        # Same checkpoint => reuse to save RAM. Adapter still operates as A -> B.
        print("[note] Model A == Model B; reusing in-memory weights.")
        model_b, tok_b = model_a, tok_a
    else:
        model_b, tok_b = load_frozen(DEFAULT_MODEL_B)

    d_a = model_a.config.hidden_size
    d_b = model_b.config.hidden_size
    print(f"\n[dims] d_A={d_a}  d_B={d_b}")

    print("\n=== Step 3: Encode 'Hello world.' through Model A ===")
    hidden_a = encode_to_hidden(model_a, tok_a, "Hello world.")
    print(f"[hidden_a] shape={tuple(hidden_a.shape)} dtype={hidden_a.dtype}")
    assert hidden_a.shape == (1, d_a), f"unexpected shape {hidden_a.shape}"

    print("\n=== Step 4: Random orthogonal adapter A -> B ===")
    adapter = Adapter(d_a, d_b).to(device).to(hidden_a.dtype)
    init_orthogonal(adapter)
    with torch.no_grad():
        hidden_b = adapter(hidden_a)
    print(f"[hidden_b] shape={tuple(hidden_b.shape)} dtype={hidden_b.dtype}")
    assert hidden_b.shape == (1, d_b), f"unexpected shape {hidden_b.shape}"

    print("\n=== Step 5: Decode through Model B ===")
    text, debug = decode_from_hidden(model_b, tok_b, hidden_b, max_new_tokens=30)
    print(f"[debug] {debug}")
    print(f"[generated] {text!r}")
    print("(garbage is expected — the adapter is untrained random orthogonal)")

    print("\n=== Step 6: Gradient sanity ===")
    # Re-enable gradients on the adapter only. Models stay frozen.
    for p in adapter.parameters():
        p.requires_grad_(True)

    # Re-encode A under no_grad (we never want grads on A's weights).
    hidden_a_ng = encode_to_hidden(model_a, tok_a, "Gradient probe sentence.")
    assert not hidden_a_ng.requires_grad, "Model A leaked grads"

    # Adapter forward must produce a leaf with grad.
    hidden_b_grad = adapter(hidden_a_ng)
    assert hidden_b_grad.requires_grad, "Adapter output did not require grad"

    # Push through Model B (frozen) and compute a dummy scalar loss.
    # We mimic decode_from_hidden's input construction but without generate(),
    # because generate() does not propagate grads.
    from src.models import _split_chat_template  # local import to avoid cycle noise

    pre, post = _split_chat_template(tok_b, "You received a signal:\n")
    pre_ids = tok_b(pre, return_tensors="pt", add_special_tokens=False).input_ids.to(
        model_b.device
    )
    post_ids = tok_b(post, return_tensors="pt", add_special_tokens=False).input_ids.to(
        model_b.device
    )
    embed = model_b.get_input_embeddings()
    pre_embeds = embed(pre_ids)
    post_embeds = embed(post_ids)
    vec_embed = hidden_b_grad.to(pre_embeds.dtype).reshape(1, 1, -1)
    inputs_embeds = torch.cat([pre_embeds, vec_embed, post_embeds], dim=1)
    attn = torch.ones(
        inputs_embeds.shape[:2], dtype=torch.long, device=model_b.device
    )

    out = model_b(inputs_embeds=inputs_embeds, attention_mask=attn)
    # Dummy loss: log-prob of token 0 at the last position. Any differentiable
    # scalar tied to B's output works.
    loss = out.logits[:, -1, :].float().log_softmax(dim=-1)[:, 0].mean()
    print(f"[dummy loss] {loss.item():.6f}")
    loss.backward()

    g = adapter.linear.weight.grad
    assert g is not None, "no gradient on adapter.linear.weight"
    finite = torch.isfinite(g).all().item()
    nonzero = (g.abs().sum() > 0).item()
    print(
        f"[grad] shape={tuple(g.shape)}  finite={finite}  "
        f"nonzero={nonzero}  norm={g.float().norm().item():.6f}"
    )
    assert finite and nonzero, "gradient is degenerate"

    print("\nPhase 1 sanity check PASSED.")


if __name__ == "__main__":
    main()
