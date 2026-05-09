"""Model loading and hidden-state I/O for the Phase 1 plumbing test.

Surface used by the sanity notebook:
    load_frozen(name)        -> (model, tokenizer)
    encode_to_hidden(...)    -> [1, d_model] tensor (last layer, last position)
    decode_from_hidden(...)  -> generated text after injecting one fake-token embed
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_A = "Qwen/Qwen3-1.7B"
DEFAULT_MODEL_B = "Qwen/Qwen3-1.7B"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    if device in ("cuda", "mps"):
        return torch.bfloat16
    return torch.float32


def load_frozen(
    name: str,
    device: str | None = None,
    dtype: torch.dtype | None = None,
):
    device = device or pick_device()
    dtype = dtype or pick_dtype(device)
    tok = AutoTokenizer.from_pretrained(name)
    # transformers v5 renamed `torch_dtype` -> `dtype`. We need v4.45+ per spec
    # but uv resolved v5; use the new name.
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    d_model = model.config.hidden_size
    print(
        f"[load_frozen] {name}  device={device}  dtype={dtype}  "
        f"params={n_params/1e9:.2f}B  d_model={d_model}"
    )
    return model, tok


@torch.no_grad()
def encode_to_hidden(
    model,
    tok,
    text: str,
    layer: int = -1,
    position: int = -1,
) -> torch.Tensor:
    """Forward pass, return hidden state at (layer, position). Shape: [1, d_model]."""
    inputs = tok(text, return_tensors="pt").to(model.device)
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    return outputs.hidden_states[layer][:, position, :]


def _split_chat_template(tok, user_text: str, sentinel: str = "<<<VEC>>>"):
    """Render the chat template with a sentinel where the vector embed will go,
    then split on it. Returns (pre_text, post_text)."""
    full = tok.apply_chat_template(
        [{"role": "user", "content": user_text + sentinel}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if sentinel not in full:
        raise RuntimeError(
            f"Sentinel {sentinel!r} not found in rendered chat template. "
            f"Tokenizer may have escaped or stripped it."
        )
    pre, post = full.split(sentinel, 1)
    return pre, post


def decode_from_hidden(
    model,
    tok,
    vec: torch.Tensor,
    user_prefix: str = "You received a signal:\n",
    max_new_tokens: int = 50,
    do_sample: bool = False,
) -> tuple[str, dict]:
    """Inject `vec` as one fake-token embedding inside Model B's chat context.

    Pipeline:
      1. Render chat template with a sentinel marking the vector slot.
      2. Tokenize and embed the pre/post text via model's input embeddings.
      3. Concat [pre_embeds, vec, post_embeds] and pass as `inputs_embeds`.
      4. Generate; decode the returned tokens.

    Returns (generated_text, debug_info).
    """
    pre, post = _split_chat_template(tok, user_prefix)
    pre_ids = tok(pre, return_tensors="pt", add_special_tokens=False).input_ids.to(
        model.device
    )
    post_ids = tok(post, return_tensors="pt", add_special_tokens=False).input_ids.to(
        model.device
    )

    embed = model.get_input_embeddings()
    pre_embeds = embed(pre_ids)
    vec_embed = vec.to(pre_embeds.dtype).to(model.device).reshape(1, 1, -1)
    post_embeds = embed(post_ids)

    if vec_embed.shape[-1] != pre_embeds.shape[-1]:
        raise ValueError(
            f"vec dim {vec_embed.shape[-1]} != model B d_model {pre_embeds.shape[-1]}. "
            f"Adapter output dim does not match Model B."
        )

    inputs_embeds = torch.cat([pre_embeds, vec_embed, post_embeds], dim=1)
    attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=model.device)

    out = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    debug = {
        "pre_len": pre_ids.shape[1],
        "post_len": post_ids.shape[1],
        "inputs_embeds_shape": tuple(inputs_embeds.shape),
        "generated_token_count": out.shape[1],
    }
    return text, debug
