"""Phase 2 diagnostic: is Qwen3 1.7B genuinely incapable, or is the dialog
protocol artificially suppressing performance?

Two checks, both on the first 20 test episodes:

  Check 1: SOLO upper bound.
    A single model sees a self-contained prompt with key + ciphertext (key
    formatted as 'a -> X' lines for clarity) and is asked for plaintext.
    If accuracy is high, the model is capable and the dialog baseline is
    measuring protocol overhead, not model capacity.

  Check 2: TWO-SHOT dialog.
    Same dialog as the baseline but B's system prompt now includes one
    worked example showing how to use the 26-char key string to decode.
    If accuracy jumps vs the smoke-test baseline, prompt engineering
    explains the failure and a "stripped" text baseline would be unfair
    to compare against the adapter.

Run:
    uv run python scripts/sanity_checks.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.text_chat import (  # noqa: E402
    A_SYSTEM,
    B_SYSTEM,
    chat_generate,
    parse_plaintext,
    run_episode,
)
from src.models import DEFAULT_MODEL_A, load_frozen, pick_device  # noqa: E402
from src.tasks.cipher_decode import char_accuracy, read_jsonl  # noqa: E402

DATA_DIR = ROOT / "data"
N_EPISODES = 20


SOLO_PROMPT_TEMPLATE = """\
Here is a substitution cipher key. Each line shows how a plaintext letter \
encodes to a ciphertext letter (left side is plaintext, right side is ciphertext):

{key_lines}

Use this key to decode the ciphertext below. To decode a single ciphertext \
letter, find which plaintext letter on the LEFT encodes to it on the right, \
then output that plaintext letter.

Ciphertext: {ciphertext}

Output ONLY the decoded plaintext. Lowercase letters only, no quotes, \
no explanation, no prefix."""


B_SYSTEM_WITH_EXAMPLE = (
    B_SYSTEM
    + "\n\n"
    + (
        "WORKED EXAMPLE so you understand the format:\n"
        "Suppose your key string is 'bcadefghijklmnopqrstuvwxyz'. "
        "This means: 'a' (alphabet position 0) encodes to key[0] = 'b'; "
        "'b' encodes to key[1] = 'c'; 'c' encodes to key[2] = 'a'; "
        "and 'd'..'z' encode to themselves.\n"
        "To DECODE a ciphertext letter, find that letter inside the key string "
        "and read off its alphabet position:\n"
        "  - ciphertext 'b' is found at key position 0, so plaintext is the 0th "
        "alphabet letter = 'a'\n"
        "  - ciphertext 'c' is found at key position 1, so plaintext = 'b'\n"
        "  - ciphertext 'a' is found at key position 2, so plaintext = 'c'\n"
        "So if A says the ciphertext is 'bca', the decoded plaintext is 'abc'.\n"
        "Apply this exact procedure to YOUR actual key when A gives you ciphertext."
    )
)


def format_key_lines(key: str) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return "\n".join(f"{alphabet[i]} -> {key[i]}" for i in range(26))


def run_solo(model, tok, episodes, max_new_tokens=50) -> list[dict]:
    results = []
    for i, ep in enumerate(episodes):
        prompt = SOLO_PROMPT_TEMPLATE.format(
            key_lines=format_key_lines(ep["key"]),
            ciphertext=ep["ciphertext"],
        )
        messages = [{"role": "user", "content": prompt}]
        t0 = time.time()
        raw, n_tok = chat_generate(
            model, tok, messages,
            max_new_tokens=max_new_tokens, enable_thinking=False,
        )
        elapsed = time.time() - t0
        pred = parse_plaintext(raw)
        results.append({
            "episode_id": ep["id"],
            "plaintext": ep["plaintext"],
            "ciphertext": ep["ciphertext"],
            "key": ep["key"],
            "raw_output": raw,
            "predicted": pred,
            "char_accuracy": char_accuracy(pred, ep["plaintext"]),
            "exact_match": pred == ep["plaintext"],
            "tokens": n_tok,
            "elapsed": elapsed,
        })
        last = results[-1]
        running_acc = statistics.mean(r["char_accuracy"] for r in results)
        print(
            f"  [solo {i+1:2d}/{len(episodes)}] "
            f"pt={ep['plaintext']!r:>17}  pred={pred!r:>17}  "
            f"acc={last['char_accuracy']:.2f}  em={int(last['exact_match'])}  "
            f"tok={n_tok}  t={elapsed:4.1f}s  running_acc={running_acc:.3f}"
        )
    return results


def run_twoshot(model_a, tok_a, model_b, tok_b, episodes) -> list[dict]:
    results = []
    for i, ep in enumerate(episodes):
        t0 = time.time()
        r = run_episode(
            model_a, tok_a, model_b, tok_b, ep,
            n_exchanges=3,
            enable_thinking=False,
            b_system=B_SYSTEM_WITH_EXAMPLE,
        )
        elapsed = time.time() - t0
        r["elapsed"] = elapsed
        results.append(r)
        running_acc = statistics.mean(rr["char_accuracy"] for rr in results)
        print(
            f"  [twoshot {i+1:2d}/{len(episodes)}] "
            f"pt={ep['plaintext']!r:>17}  pred={r['predicted_plaintext']!r:>17}  "
            f"acc={r['char_accuracy']:.2f}  em={int(r['exact_match'])}  "
            f"tok={r['total_tokens']}  t={elapsed:4.1f}s  running_acc={running_acc:.3f}"
        )
    return results


def summarize(name, results) -> dict:
    accs = [r["char_accuracy"] for r in results]
    ems = [r["exact_match"] for r in results]
    s = {
        "name": name,
        "n": len(results),
        "mean_char_accuracy": statistics.mean(accs),
        "stdev_char_accuracy": statistics.stdev(accs) if len(accs) > 1 else 0.0,
        "exact_match_rate": statistics.mean(ems),
    }
    print(f"\n[{name}] n={s['n']}  mean_acc={s['mean_char_accuracy']:.3f}  "
          f"stdev={s['stdev_char_accuracy']:.3f}  em_rate={s['exact_match_rate']:.3f}")
    return s


def main() -> None:
    device = pick_device()
    print(f"[device] {device}\n")

    test_path = DATA_DIR / "cipher_test.jsonl"
    episodes = list(read_jsonl(test_path))[:N_EPISODES]
    print(f"[load] Model: {DEFAULT_MODEL_A}")
    model, tok = load_frozen(DEFAULT_MODEL_A)

    print(f"\n=== Check 1: SOLO upper bound (n={len(episodes)}) ===")
    solo_results = run_solo(model, tok, episodes)
    solo_summary = summarize("solo", solo_results)

    print(f"\n=== Check 2: TWO-SHOT dialog (n={len(episodes)}) ===")
    twoshot_results = run_twoshot(model, tok, model, tok, episodes)
    twoshot_summary = summarize("twoshot_dialog", twoshot_results)

    out_dir = ROOT / "experiments" / "run_phase2_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "solo.jsonl").write_text(
        "\n".join(json.dumps(r) for r in solo_results) + "\n"
    )
    (out_dir / "twoshot.jsonl").write_text(
        "\n".join(json.dumps(r) for r in twoshot_results) + "\n"
    )
    (out_dir / "summary.json").write_text(
        json.dumps({"solo": solo_summary, "twoshot": twoshot_summary}, indent=2)
    )
    print(f"\n[done] Outputs in {out_dir}")


if __name__ == "__main__":
    main()
