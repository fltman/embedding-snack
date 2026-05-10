"""Text-only chat baseline for the cipher-decode task.

Runs Player A and Player B as two separate chat sessions, alternating turns.
Both sides use the same chat template config (`enable_thinking=False`) to
keep the comparison with the adapter condition apples-to-apples.

Per Phase 2 spec: each episode gets `n_exchanges` rounds (default 3), then
B is explicitly prompted for the plaintext.
"""
from __future__ import annotations

import re

import torch

from src.tasks.cipher_decode import char_accuracy

A_SYSTEM = (
    "You are Player A in a cooperative cipher-decoding game with Player B. "
    "You see only the CIPHERTEXT. Player B sees only the cipher KEY "
    "(a 26-character permutation of the alphabet a-z, where the i-th character "
    "is the image of the i-th letter, i.e. key[0] is the encoding of 'a'). "
    "Cooperate with Player B to recover the plaintext. "
    "Keep messages short. Do not output the final plaintext yourself — "
    "Player B will produce it."
)

B_SYSTEM = (
    "You are Player B in a cooperative cipher-decoding game with Player A. "
    "You see only the cipher KEY (a 26-character permutation of a-z, where "
    "the i-th character is the image of the i-th letter — so to DECODE, "
    "find each ciphertext character in the key and read off the position). "
    "Player A sees only the ciphertext. "
    "Cooperate with Player A to recover the plaintext. "
    "Keep messages short. When explicitly asked for the plaintext at the end, "
    "output ONLY the plaintext, lowercase letters only, no quotes, no explanation."
)

FINAL_PROMPT = (
    "Now output ONLY the decoded plaintext. Lowercase letters only. "
    "No quotes. No prefix. No explanation. Just the plaintext."
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_LETTERS_RE = re.compile(r"[a-z]+")


def _apply_template(tok, messages, enable_thinking: bool = False) -> str:
    """Render chat messages, trying to honor enable_thinking if the tokenizer
    supports it (Qwen3). Falls back to plain template otherwise."""
    try:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def chat_generate(
    model,
    tok,
    messages: list[dict],
    max_new_tokens: int = 200,
    enable_thinking: bool = False,
) -> tuple[str, int]:
    """Run one assistant turn and return (text, generated_token_count)."""
    text = _apply_template(tok, messages, enable_thinking=enable_thinking)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    input_len = enc.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    new_tokens = out[0, input_len:]
    decoded = tok.decode(new_tokens, skip_special_tokens=True)
    return decoded, int(new_tokens.shape[0])


def parse_plaintext(text: str) -> str:
    """Heuristic extractor for B's final message.

    1. Strip any <think>...</think> block (in case thinking leaks through).
    2. Lowercase, trim.
    3. Find all runs of [a-z]+. Prefer runs of plausible plaintext length
       (3-20). Among those, prefer the longest. If nothing in range, take
       the last run. If no letters at all, return "".
    """
    text = _THINK_RE.sub("", text).strip().lower()
    runs = _LETTERS_RE.findall(text)
    if not runs:
        return ""
    in_range = [r for r in runs if 3 <= len(r) <= 20]
    if in_range:
        return max(in_range, key=len)
    return runs[-1]


def run_episode(
    model_a,
    tok_a,
    model_b,
    tok_b,
    episode: dict,
    n_exchanges: int = 3,
    max_message_tokens: int = 160,
    max_final_tokens: int = 40,
    enable_thinking: bool = False,
    a_system: str = A_SYSTEM,
    b_system: str = B_SYSTEM,
) -> dict:
    """Run a single episode through the text-chat baseline."""
    ciphertext = episode["ciphertext"]
    key = episode["key"]
    plaintext = episode["plaintext"]

    a_history = [
        {"role": "system", "content": a_system},
        {"role": "user", "content": f"The ciphertext is: {ciphertext}\n\nTalk to Player B."},
    ]
    b_history = [
        {"role": "system", "content": b_system},
        {"role": "user", "content": f"The cipher key is: {key}\n\nTalk to Player A."},
    ]

    transcript: list[dict] = []
    a_tokens = 0
    b_tokens = 0

    for round_idx in range(n_exchanges):
        msg_a, n_a = chat_generate(
            model_a, tok_a, a_history,
            max_new_tokens=max_message_tokens, enable_thinking=enable_thinking,
        )
        a_history.append({"role": "assistant", "content": msg_a})
        b_history.append({"role": "user", "content": msg_a})
        transcript.append({"round": round_idx, "speaker": "A", "text": msg_a, "tokens": n_a})
        a_tokens += n_a

        msg_b, n_b = chat_generate(
            model_b, tok_b, b_history,
            max_new_tokens=max_message_tokens, enable_thinking=enable_thinking,
        )
        b_history.append({"role": "assistant", "content": msg_b})
        a_history.append({"role": "user", "content": msg_b})
        transcript.append({"round": round_idx, "speaker": "B", "text": msg_b, "tokens": n_b})
        b_tokens += n_b

    b_history.append({"role": "user", "content": FINAL_PROMPT})
    final_raw, n_final = chat_generate(
        model_b, tok_b, b_history,
        max_new_tokens=max_final_tokens, enable_thinking=enable_thinking,
    )
    b_tokens += n_final
    pred = parse_plaintext(final_raw)

    return {
        "episode_id": episode["id"],
        "plaintext": plaintext,
        "ciphertext": ciphertext,
        "key": key,
        "transcript": transcript,
        "final_b_raw": final_raw,
        "predicted_plaintext": pred,
        "char_accuracy": char_accuracy(pred, plaintext),
        "exact_match": pred == plaintext,
        "a_tokens": a_tokens,
        "b_tokens": b_tokens,
        "total_tokens": a_tokens + b_tokens,
        "n_exchanges": n_exchanges,
    }
