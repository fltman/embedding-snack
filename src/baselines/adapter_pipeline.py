"""Adapter-mediated cipher-decode pipeline.

A receives ciphertext as text, runs a forward pass, exposes the last-layer
hidden state at the final token position. The adapter projects A's hidden
state into B's embedding space. B receives:
  - system prompt with abstract role description (no key value)
  - user message 1: alphabet-aligned decoding-key table + framing
  - user message 2: a short text prefix + the adapter-projected vector as a
    single fake-token embedding + a closing prompt asking for the answer
    in the <answer>...</answer> protocol.

B then generates text. We parse the answer tag and score against the
ground-truth plaintext.

Phase 3 uses an untrained random orthogonal adapter and runs single-shot
(one A→B vector per episode). Phase 4 will train the same adapter and may
extend to multi-turn (N=3) hidden-state exchanges.
"""
from __future__ import annotations

import torch

from src.baselines.text_chat import (
    A_SYSTEM,
    B_SYSTEM_DECODING,
    B_SYSTEM_ENCODING,
    format_key_table,
)
from src.models import encode_to_hidden
from src.tasks.cipher_decode import (
    char_accuracy,
    parse_answer,
    tokens_before_answer,
)


SIGNAL_SENTINEL = "<<<SIGNAL>>>"


def _build_b_user_init(key: str, key_direction: str) -> str:
    """Same content as run_episode in text_chat.py — key table + framing.

    Kept aligned with `phase-2-dialog-tabkey-87pct` so the adapter pipeline
    can be compared directly against the text-dialog baseline. If text_chat's
    init changes, this should change in lockstep.
    """
    if key_direction == "decoding":
        return (
            "Here is a substitution cipher decoding table. Each line maps a CIPHERTEXT "
            "letter to its PLAINTEXT decoding (left side is ciphertext, right side is plaintext):\n\n"
            f"{format_key_table(key, key_direction)}\n\n"
            "Use this table to decode the ciphertext that Player A will signal. "
            "For each ciphertext letter, look it up on the LEFT and read off the plaintext letter on the right.\n\n"
            "Player A will now send you a signal."
        )
    return (
        "Here is a substitution cipher encoding table. Each line maps a PLAINTEXT "
        "letter to its CIPHERTEXT encoding (left side is plaintext, right side is ciphertext):\n\n"
        f"{format_key_table(key, key_direction)}\n\n"
        "Use this table to decode the ciphertext that Player A will signal. "
        "For each ciphertext letter, find it on the RIGHT and output the matching plaintext letter from the LEFT.\n\n"
        "Player A will now send you a signal."
    )


SIGNAL_USER_PREFIX = "Player A's signal: "
SIGNAL_USER_SUFFIX = (
    "\n\nThat signal carries information about the ciphertext. Use it together "
    "with your decoding table to recover the plaintext.\n\n"
    "On the FINAL LINE, output exactly:\n"
    "<answer>YOUR_PLAINTEXT_HERE</answer>\n"
    "Replace YOUR_PLAINTEXT_HERE with the decoded plaintext (lowercase letters only)."
)


def _apply_template(tok, messages, enable_thinking: bool = False) -> str:
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def chat_generate_with_signal(
    model_b,
    tok_b,
    b_system: str,
    b_user_init: str,
    signal_vec: torch.Tensor,
    max_new_tokens: int = 500,
    enable_thinking: bool = False,
) -> tuple[str, int, int]:
    """Run B with system + user-init + (signal-bearing user message) and generate.

    Returns (decoded_text, total_generated_tokens, input_embed_length).
    """
    user_with_signal = (
        SIGNAL_USER_PREFIX + SIGNAL_SENTINEL + SIGNAL_USER_SUFFIX
    )
    messages = [
        {"role": "system", "content": b_system},
        {"role": "user", "content": b_user_init},
        {"role": "user", "content": user_with_signal},
    ]
    full = _apply_template(tok_b, messages, enable_thinking=enable_thinking)
    if SIGNAL_SENTINEL not in full:
        raise RuntimeError(
            f"Signal sentinel {SIGNAL_SENTINEL!r} missing from rendered template."
        )
    pre, post = full.split(SIGNAL_SENTINEL, 1)

    pre_ids = tok_b(pre, return_tensors="pt", add_special_tokens=False).input_ids.to(model_b.device)
    post_ids = tok_b(post, return_tensors="pt", add_special_tokens=False).input_ids.to(model_b.device)

    embed = model_b.get_input_embeddings()
    pre_embeds = embed(pre_ids)
    post_embeds = embed(post_ids)

    vec_embed = signal_vec.to(pre_embeds.dtype).to(model_b.device).reshape(1, 1, -1)
    if vec_embed.shape[-1] != pre_embeds.shape[-1]:
        raise ValueError(
            f"Adapter output dim {vec_embed.shape[-1]} != Model B d_model {pre_embeds.shape[-1]}"
        )

    inputs_embeds = torch.cat([pre_embeds, vec_embed, post_embeds], dim=1)
    attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=model_b.device)
    input_len = inputs_embeds.shape[1]

    with torch.no_grad():
        out = model_b.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok_b.eos_token_id,
        )
    text = tok_b.decode(out[0], skip_special_tokens=True)
    return text, int(out.shape[1]), input_len


def encode_a_signal(
    model_a,
    tok_a,
    ciphertext: str,
    layer: int = -1,
    position: int = -1,
) -> torch.Tensor:
    """Run A on the ciphertext and return its hidden state at (layer, position).

    Phase 3 uses a minimal A-prompt: just `"The ciphertext is: <ct>"`. This
    keeps A's role implicit; we're not asking A to do anything beyond letting
    its forward pass register the ciphertext in its representation.
    """
    a_text = f"The ciphertext is: {ciphertext}"
    return encode_to_hidden(model_a, tok_a, a_text, layer=layer, position=position)


def run_episode_adapter(
    model_a,
    tok_a,
    model_b,
    tok_b,
    adapter,
    episode: dict,
    key_direction: str = "decoding",
    max_new_tokens: int = 500,
    enable_thinking: bool = False,
    a_layer: int = -1,
    a_position: int = -1,
) -> dict:
    """Single-shot A → adapter → B → text → score for one episode."""
    plaintext = episode["plaintext"]
    ciphertext = episode["ciphertext"]
    key = episode["key"]

    # Step 1: A's hidden state at end of "ciphertext" prompt.
    hidden_a = encode_a_signal(
        model_a, tok_a, ciphertext, layer=a_layer, position=a_position,
    )

    # Step 2: adapter projects.
    with torch.no_grad():
        signal = adapter(hidden_a)

    # Step 3: B has key in user-init, signal as fake-token embedding, asks for answer.
    b_system = (
        B_SYSTEM_DECODING if key_direction == "decoding" else B_SYSTEM_ENCODING
    )
    b_user_init = _build_b_user_init(key, key_direction)
    raw, n_tok, in_len = chat_generate_with_signal(
        model_b, tok_b, b_system, b_user_init, signal,
        max_new_tokens=max_new_tokens, enable_thinking=enable_thinking,
    )

    parsed, found = parse_answer(raw)
    pred = parsed or ""
    n_before = tokens_before_answer(tok_b, raw)

    return {
        "ep": episode["id"],
        "ciphertext": ciphertext,
        "plaintext_gt": plaintext,
        "key": key,
        "key_direction": key_direction,
        "raw_output": raw,
        "answer_tag_found": found,
        "predicted_plaintext": pred,
        "char_acc": char_accuracy(pred, plaintext),
        "exact_match": pred == plaintext,
        "tokens_used": n_tok,
        "tokens_used_before_answer_tag": n_before,
        "input_embed_length": in_len,
        "signal_norm": float(signal.detach().float().norm().item()),
    }
