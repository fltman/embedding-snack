"""Text-only chat baseline for the cipher-decode task.

Runs Player A and Player B as two separate chat sessions, alternating turns.
Both sides use the same chat template config (`enable_thinking=False`) to
keep the comparison with the adapter condition apples-to-apples.

Per Phase 2 spec: each episode gets `n_exchanges` rounds (default 3), then
B is explicitly prompted for the plaintext via the same `<answer>` tag
protocol used in the solo sanity check.

Key presentation: B's system prompt contains the cipher key as an explicit
alphabet-aligned table (`a -> X`, `b -> Y`, ...), not as a flat string. The
flat-string presentation (run_006) caused FM1 (positional-index lookup
error). The table format removes the ambiguity by binding each entry to a
specific alphabet letter.
"""
from __future__ import annotations

import torch

from src.tasks.cipher_decode import (
    alphabet_for_key,
    char_accuracy,
    parse_answer,
    tokens_before_answer,
)

A_SYSTEM = (
    "You are Player A in a cooperative cipher-decoding game with Player B. "
    "You see only the CIPHERTEXT. Player B has the cipher key. "
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

# Backwards-compat alias used by the two-shot sanity check.
B_SYSTEM = B_SYSTEM_ENCODING

FINAL_PROMPT = (
    "Now produce the final answer. You may show brief work first, then on the FINAL LINE "
    "output exactly:\n"
    "<answer>YOUR_PLAINTEXT_HERE</answer>\n"
    "Replace YOUR_PLAINTEXT_HERE with the decoded plaintext (lowercase letters only)."
)


def format_key_table(key: str, key_direction: str = "decoding") -> str:
    """Render the key as an alphabet-aligned table.

    For `decoding` direction: `key[i]` is the plaintext for ciphertext alphabet[i],
        rendered as `{alphabet[i]} -> {key[i]}` (left=ciphertext, right=plaintext).
    For `encoding` direction: `key[i]` is the ciphertext for plaintext alphabet[i],
        rendered as `{alphabet[i]} -> {key[i]}` (left=plaintext, right=ciphertext).
    Same line format; the surrounding text disambiguates the meaning.
    """
    alphabet = alphabet_for_key(key)
    return "\n".join(f"{alphabet[i]} -> {key[i]}" for i in range(len(alphabet)))


def make_b_system(key: str, key_direction: str = "decoding") -> str:
    """Diagnostic helper used by sanity_checks.py --mode system-key.

    Builds a B system prompt that embeds the alphabet-aligned key table directly.
    NOT used by run_episode (the dialog baseline keeps the key in B's first
    user message, matching run_005/run_006 placement).
    """
    if key_direction == "decoding":
        body = (
            "You are Player B in a cooperative cipher-decoding game with Player A.\n\n"
            "You have a DECODING table that maps each ciphertext letter to its plaintext:\n\n"
            f"{format_key_table(key, key_direction)}\n\n"
            "To DECODE a ciphertext letter, find it on the LEFT side of the table and "
            "output the corresponding letter on the RIGHT.\n\n"
            "Player A sees only the ciphertext (you do not see it directly until they tell you).\n"
            "Cooperate with Player A to recover the plaintext.\n\n"
            "Keep messages short. When asked for the final answer, output it on the FINAL LINE as exactly:\n"
            "<answer>YOUR_PLAINTEXT_HERE</answer>"
        )
    else:
        body = (
            "You are Player B in a cooperative cipher-decoding game with Player A.\n\n"
            "You have an ENCODING table that maps each plaintext letter to its ciphertext:\n\n"
            f"{format_key_table(key, key_direction)}\n\n"
            "To DECODE a ciphertext letter, find it on the RIGHT side of the table and "
            "output the corresponding letter on the LEFT.\n\n"
            "Player A sees only the ciphertext (you do not see it directly until they tell you).\n"
            "Cooperate with Player A to recover the plaintext.\n\n"
            "Keep messages short. When asked for the final answer, output it on the FINAL LINE as exactly:\n"
            "<answer>YOUR_PLAINTEXT_HERE</answer>"
        )
    return body


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
        b_system = B_SYSTEM_DECODING if key_direction == "decoding" else B_SYSTEM_ENCODING

    # B's user-init carries the key as an alphabet-aligned table — same content
    # block as run_005 SOLO_PROMPT_DECODING's framing (Block 1), with structurally
    # required adaptations: ciphertext comes from A (not "below"), and the
    # answer-tag instruction is in FINAL_PROMPT (not repeated here).
    if key_direction == "decoding":
        b_user_init = (
            "Here is a substitution cipher decoding table. Each line maps a CIPHERTEXT "
            "letter to its PLAINTEXT decoding (left side is ciphertext, right side is plaintext):\n\n"
            f"{format_key_table(key, key_direction)}\n\n"
            "Use this table to decode the ciphertext that Player A will provide. "
            "For each ciphertext letter, look it up on the LEFT and read off the plaintext letter on the right.\n\n"
            "Talk to Player A."
        )
    else:
        b_user_init = (
            "Here is a substitution cipher encoding table. Each line maps a PLAINTEXT "
            "letter to its CIPHERTEXT encoding (left side is plaintext, right side is ciphertext):\n\n"
            f"{format_key_table(key, key_direction)}\n\n"
            "Use this table to decode the ciphertext that Player A will provide. "
            "For each ciphertext letter, find it on the RIGHT and output the matching plaintext letter from the LEFT.\n\n"
            "Talk to Player A."
        )

    a_history = [
        {"role": "system", "content": a_system},
        {"role": "user", "content": f"The ciphertext is: {ciphertext}\n\nTalk to Player B."},
    ]
    b_history = [
        {"role": "system", "content": b_system},
        {"role": "user", "content": b_user_init},
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
