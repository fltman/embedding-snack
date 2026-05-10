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

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.text_chat import (  # noqa: E402
    A_SYSTEM,
    B_SYSTEM,
    chat_generate,
    run_episode,
)
from src.models import DEFAULT_MODEL_A, DEFAULT_MODEL_B, load_pair, pick_device  # noqa: E402
from src.tasks.cipher_decode import (  # noqa: E402
    alphabet_for_key,
    char_accuracy,
    parse_answer as _parse_answer_shared,
    read_jsonl,
    tokens_before_answer as _tokens_before_answer_shared,
)

DATA_DIR = ROOT / "data"
N_EPISODES = 20


SOLO_PROMPT_ENCODING = """\
Here is a substitution cipher key. Each line shows how a plaintext letter \
encodes to a ciphertext letter (left side is plaintext, right side is ciphertext):

{key_lines}

Use this key to decode the ciphertext below. To decode a single ciphertext \
letter, find which plaintext letter on the LEFT encodes to it on the right, \
then output that plaintext letter.

Ciphertext: {ciphertext}

You may show your work if helpful. Then on the FINAL LINE, output exactly:
<answer>YOUR_PLAINTEXT_HERE</answer>

Replace YOUR_PLAINTEXT_HERE with the decoded plaintext (lowercase a-z only, \
no spaces, no punctuation). Do not put anything after the closing tag."""


SOLO_PROMPT_DECODING = """\
Here is a substitution cipher decoding table. Each line maps a CIPHERTEXT \
letter to its PLAINTEXT decoding (left side is ciphertext, right side is plaintext):

{key_lines}

Use this table to decode the ciphertext below. For each ciphertext letter, \
look it up on the LEFT and read off the plaintext letter on the right.

Ciphertext: {ciphertext}

You may show your work if helpful. Then on the FINAL LINE, output exactly:
<answer>YOUR_PLAINTEXT_HERE</answer>

Replace YOUR_PLAINTEXT_HERE with the decoded plaintext (lowercase a-z only, \
no spaces, no punctuation). Do not put anything after the closing tag."""


# Re-exported for backwards compatibility — definitions live in src.tasks.cipher_decode.
parse_answer = _parse_answer_shared
tokens_before_answer = _tokens_before_answer_shared


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
    """Render `alphabet[i] -> key[i]` lines, one per row.

    Interpretation depends on `key_direction` of the dataset:
      - encoding: key[i] is the ciphertext for plaintext alphabet[i].
      - decoding: key[i] is the plaintext for ciphertext alphabet[i].
    The text formatting is identical; only the prompt wording differs.
    """
    alphabet = alphabet_for_key(key)
    return "\n".join(f"{alphabet[i]} -> {key[i]}" for i in range(len(alphabet)))


def run_solo(model, tok, episodes, max_new_tokens=500, key_direction: str = "encoding") -> list[dict]:
    template = SOLO_PROMPT_ENCODING if key_direction == "encoding" else SOLO_PROMPT_DECODING
    results = []
    for i, ep in enumerate(episodes):
        prompt = template.format(
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
        parsed, found = parse_answer(raw)
        n_before = tokens_before_answer(tok, raw)
        pt_pred = parsed or ""
        results.append({
            "ep": ep["id"],
            "ciphertext": ep["ciphertext"],
            "plaintext_gt": ep["plaintext"],
            "key": ep["key"],
            "raw_output": raw,
            "answer_tag_found": found,
            "parsed_plaintext": pt_pred,
            "char_acc": char_accuracy(pt_pred, ep["plaintext"]),
            "exact_match": pt_pred == ep["plaintext"],
            "tokens_used": n_tok,
            "tokens_used_before_answer_tag": n_before,
            "elapsed": elapsed,
        })
        last = results[-1]
        running_acc = statistics.mean(r["char_acc"] for r in results)
        running_compliance = statistics.mean(int(r["answer_tag_found"]) for r in results)
        print(
            f"  [solo {i+1:2d}/{len(episodes)}] "
            f"pt={ep['plaintext']!r:>17}  pred={pt_pred!r:>17}  "
            f"tag={'Y' if found else 'N'}  "
            f"acc={last['char_acc']:.2f}  em={int(last['exact_match'])}  "
            f"tok={n_tok}  pre_tag={n_before}  t={elapsed:5.1f}s  "
            f"running_acc={running_acc:.3f}  compliance={running_compliance:.2f}"
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
    accs = [r.get("char_acc", r.get("char_accuracy", 0.0)) for r in results]
    ems = [r["exact_match"] for r in results]
    compliance_present = "answer_tag_found" in results[0]
    tokens_used = [r.get("tokens_used", r.get("tokens", 0)) for r in results]
    s = {
        "name": name,
        "n": len(results),
        "mean_char_acc": statistics.mean(accs),
        "stdev_char_acc": statistics.stdev(accs) if len(accs) > 1 else 0.0,
        "exact_match_rate": statistics.mean(ems),
        "mean_tokens_used": statistics.mean(tokens_used),
    }
    if compliance_present:
        compliance = [int(r["answer_tag_found"]) for r in results]
        s["format_compliance_rate"] = statistics.mean(compliance)
        # tokens_before_answer_tag only meaningful when the tag is found
        pre_tags = [r["tokens_used_before_answer_tag"] for r in results if r["tokens_used_before_answer_tag"] is not None]
        if pre_tags:
            s["mean_tokens_before_answer_tag"] = statistics.mean(pre_tags)
            s["median_tokens_before_answer_tag"] = statistics.median(pre_tags)
    print(f"\n[{name}] n={s['n']}  mean_acc={s['mean_char_acc']:.3f}  "
          f"em_rate={s['exact_match_rate']:.3f}  mean_tok={s['mean_tokens_used']:.0f}")
    if "format_compliance_rate" in s:
        print(f"           format_compliance={s['format_compliance_rate']:.2f}  "
              f"mean_tok_before_tag={s.get('mean_tokens_before_answer_tag', 0):.1f}")
    return s


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--solo-only", action="store_true", help="Skip the two-shot dialog check.")
    p.add_argument("--out-name", type=str, default="run_phase2_sanity",
                   help="Output directory under experiments/.")
    p.add_argument("--dataset", type=str, default="data/cipher_v2_test.jsonl",
                   help="Path to the JSONL test dataset (relative to repo root).")
    p.add_argument("--n", type=int, default=N_EPISODES,
                   help="How many episodes from the dataset.")
    p.add_argument("--key-direction", type=str, default="encoding",
                   choices=["encoding", "decoding"],
                   help="How the dataset's `key` field should be interpreted: "
                        "encoding (a -> X, default) or decoding (X -> a).")
    args = p.parse_args()

    device = pick_device()
    print(f"[device] {device}\n")

    test_path = ROOT / args.dataset
    episodes = list(read_jsonl(test_path))[: args.n]
    head = episodes[0]
    print(
        f"[dataset] {args.dataset}  n={len(episodes)}  alphabet_size={len(head['key'])}  "
        f"first: pt={head['plaintext']!r} ct={head['ciphertext']!r}"
    )
    model_a, tok_a, model_b, tok_b = load_pair(
        DEFAULT_MODEL_A, DEFAULT_MODEL_B,
        share_model_weights=True,  # 16 GB Mac — two separate 4B copies don't fit
    )

    print(f"\n=== Check 1: SOLO upper bound (n={len(episodes)}, key_direction={args.key_direction}) ===")
    solo_results = run_solo(model_a, tok_a, episodes, key_direction=args.key_direction)
    solo_summary = summarize("solo", solo_results)

    out_dir = ROOT / "experiments" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "solo.jsonl").write_text(
        "\n".join(json.dumps(r) for r in solo_results) + "\n"
    )

    summary_payload = {"solo": solo_summary}

    if not args.solo_only:
        print(f"\n=== Check 2: TWO-SHOT dialog (n={len(episodes)}) ===")
        twoshot_results = run_twoshot(model_a, tok_a, model_b, tok_b, episodes)
        twoshot_summary = summarize("twoshot_dialog", twoshot_results)
        (out_dir / "twoshot.jsonl").write_text(
            "\n".join(json.dumps(r) for r in twoshot_results) + "\n"
        )
        summary_payload["twoshot"] = twoshot_summary

    (out_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2))
    print(f"\n[done] Outputs in {out_dir}")


if __name__ == "__main__":
    main()
