"""Phase 2: text-chat baseline for the cipher-decode task.

Runs the text-only baseline on a subset of the test set, logs every
episode to JSONL, and writes a markdown summary.

We run on a 100-episode subset (not the full 500) because each episode
takes 7 chat-generations on a small model on MPS — full 500 would be
many hours. Phase 5 will run the full 500 against the trained adapter
for the final results table.

Run:
    uv run python scripts/02_text_baseline.py
    uv run python scripts/02_text_baseline.py --n 5    # smoke test
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.text_chat import run_episode  # noqa: E402
from src.models import DEFAULT_MODEL_A, DEFAULT_MODEL_B, load_pair, pick_device  # noqa: E402
from src.tasks.cipher_decode import read_jsonl  # noqa: E402

DATA_DIR = ROOT / "data"
EXPERIMENTS_DIR = ROOT / "experiments"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100, help="Number of test episodes to run.")
    p.add_argument("--n-exchanges", type=int, default=3, help="Rounds per episode (A+B counts as 1 round).")
    p.add_argument("--enable-thinking", action="store_true", help="Run the eval pass with Qwen3 thinking on (default off).")
    p.add_argument("--out-name", type=str, default=None, help="Override the run directory name.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.out_name or f"run_phase2_{timestamp}"
    run_dir = EXPERIMENTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    test_path = DATA_DIR / "cipher_test.jsonl"
    test = list(read_jsonl(test_path))
    test = test[: args.n]
    print(f"[config] device={device}  n={len(test)}  n_exchanges={args.n_exchanges}  "
          f"enable_thinking={args.enable_thinking}  run_dir={run_dir}")

    # Save the config used for this run.
    config = {
        "phase": 2,
        "condition": "text_baseline",
        "model_a": DEFAULT_MODEL_A,
        "model_b": DEFAULT_MODEL_B,
        "share_model_weights": True,
        "device": device,
        "n_episodes": len(test),
        "n_exchanges": args.n_exchanges,
        "enable_thinking": args.enable_thinking,
        "test_dataset": str(test_path.relative_to(ROOT)),
        "timestamp": timestamp,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    model_a, tok_a, model_b, tok_b = load_pair(
        DEFAULT_MODEL_A, DEFAULT_MODEL_B,
        share_model_weights=True,  # 16 GB Mac — two separate 4B copies don't fit
    )

    log_path = run_dir / "episodes.jsonl"
    accuracies: list[float] = []
    exacts: list[bool] = []
    total_tokens_per_ep: list[int] = []
    times: list[float] = []

    print(f"\n[run] Logging to {log_path}")
    with log_path.open("w") as logf:
        for i, ep in enumerate(test):
            t0 = time.time()
            result = run_episode(
                model_a, tok_a, model_b, tok_b, ep,
                n_exchanges=args.n_exchanges,
                enable_thinking=args.enable_thinking,
            )
            elapsed = time.time() - t0
            result["elapsed_seconds"] = elapsed
            logf.write(json.dumps(result) + "\n")
            logf.flush()

            accuracies.append(result["char_accuracy"])
            exacts.append(result["exact_match"])
            total_tokens_per_ep.append(result["total_tokens"])
            times.append(elapsed)

            running_acc = statistics.mean(accuracies)
            running_em = statistics.mean(exacts)
            print(
                f"[{i+1:3d}/{len(test)}] pt={ep['plaintext']!r:>17}  "
                f"pred={result['predicted_plaintext']!r:>17}  "
                f"acc={result['char_accuracy']:.2f}  em={int(result['exact_match'])}  "
                f"tok={result['total_tokens']:4d}  t={elapsed:5.1f}s  "
                f"running_acc={running_acc:.3f}  running_em={running_em:.3f}"
            )

    summary = {
        "n_episodes": len(test),
        "mean_char_accuracy": statistics.mean(accuracies),
        "stdev_char_accuracy": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
        "exact_match_rate": statistics.mean(exacts),
        "mean_total_tokens": statistics.mean(total_tokens_per_ep),
        "mean_elapsed_seconds": statistics.mean(times),
        "median_elapsed_seconds": statistics.median(times),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[summary] {json.dumps(summary, indent=2)}")
    print(f"\n[done] Outputs in {run_dir}")


if __name__ == "__main__":
    main()
