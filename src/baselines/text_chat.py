"""Text-only chat baseline for the cipher-decode task.

Runs Player A and Player B as two separate chat sessions, alternating turns.
Both sides use the same chat template config (`enable_thinking=False`) to
keep the comparison with the adapter condition apples-to-apples.

Per Phase 2 spec: each episode gets `n_exchanges` rounds (default 3), then
B is explicitly prompted for the plaintext via the same `<answer>` tag
protocol used in the solo sanity check.
"""
from __future__ import annotations

import torch

from src.tasks.cipher_decode import char_accuracy, parse_answer, tokens_before_answer

A_SYSTEM = (
    "You are Player A in a cooperative cipher-decoding game with Player B. "
    "You see only the CIPHERTEXT. Player B sees the cipher key. "
    "Cooperate with Player B to recover the plaintext. "
    "Keep messages short. Do not output the final plaintext yourself — "
    "Player B will produce it."
)

B_SYSTEM_ENCODING = (
    "You are Player B in a cooperative cipher-decoding game with Player A. "
    "You see only the ENCODING key: a permutation of the alphabet (lowercase letters), "
    "where key[i] is the ciphertext image of the i-th plaintext letter. "
    "To DECODE a ciphertext letter, find that letter in the key string and output "
    "the alphabet letter at that position. "
    "Player A sees only the ciphertext. "
    "Cooperate with Player A to recover the plaintext. "
    "Keep messages short. "
    "When asked for the final answer, output it on the FINAL LINE as exactly:\n"
    "<answer>YOUR_PLAINTEXT_HERE</answer>"
)

B_SYSTEM_DECODING = (
    "You are Player B in a cooperative cipher-decoding game with Player A. "
    "You see only a DECODING key: a permutation of the alphabet (lowercase letters), "
    "where the i-th character is the plaintext for the i-th ciphertext letter. "
    "To DECODE a ciphertext letter C, find C's alphabet position i and output key[i]. "
    "Player A sees only the ciphertext. "
    "Cooperate with Player A to recover the plaintext. "
    "Keep messages short. "
    "When asked for the final answer, output it on the FINAL LINE as exactly:\n"
    "<answer>YOUR_PLAINTEXT_HERE</answer>"
)

FINAL_PROMPT = (
    "Now produce the final answer. You may show brief work first, then on the FINAL LINE "
    "output exactly:\n"
    "<answer>YOUR_PLAINTEXT_HERE</answer>\n"
    "Replace YOUR_PLAINTEXT_HERE with the decoded plaintext (lowercase letters only)."
)


# Backwards-compat alias used by the older sanity_checks two-shot run.
B_SYSTEM = B_SYSTEM_ENCODING


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


def run_episode(
    model_a,
    tok_a,
    model_b,
    tok_b,
    episode: dict,
    n_exchanges: int = 3,
    max_message_tokens: int = 200,
    max_final_tokens: int = 500,
    enable_thinking: bool = False,
    key_direction: str = "encoding",
    a_system: str = A_SYSTEM,
    b_system: str | None = None,
) -> dict:
    """Run a single episode through the text-chat baseline."""
    ciphertext = episode["ciphertext"]
    key = episode["key"]
    plaintext = episode["plaintext"]

    if b_system is None:
        b_system = B_SYSTEM_ENCODING if key_direction == "encoding" else B_SYSTEM_DECODING

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
    parsed, found = parse_answer(final_raw)
    pred = parsed or ""
    n_before_tag = tokens_before_answer(tok_b, final_raw)

    return {
        "episode_id": episode["id"],
        "plaintext": plaintext,
        "ciphertext": ciphertext,
        "key": key,
        "key_direction": key_direction,
        "transcript": transcript,
        "final_b_raw": final_raw,
        "answer_tag_found": found,
        "predicted_plaintext": pred,
        "char_accuracy": char_accuracy(pred, plaintext),
        "exact_match": pred == plaintext,
        "a_tokens": a_tokens,
        "b_tokens": b_tokens,
        "total_tokens": a_tokens + b_tokens,
        "tokens_used_before_answer_tag": n_before_tag,
        "n_exchanges": n_exchanges,
    }
