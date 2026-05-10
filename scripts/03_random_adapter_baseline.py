"""Phase 3: random adapter baseline.

Runs the cipher-decode task with an untrained random orthogonal adapter
between A and B. Single-shot (one A→B vector per episode). Confirms that
the protocol learns something non-trivial only when trained — Phase 3 is
expected to land near chance accuracy (1/12 ≈ 0.083 per character on the
12-letter alphabet) and well below Phase 2's 0.872 dialog baseline.

Run:
    uv run python scripts/03_random_adapter_baseline.py --n 100 \\
        --dataset data/cipher_v3_test.jsonl --key-direction decoding \\
        --out-name run_010_random_adapter
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adapter import Adapter, init_orthogonal  # noqa: E402
from src.baselines.adapter_pipeline import run_episode_adapter  # noqa: E402
from src.models import (  # noqa: E402
    DEFAULT_MODEL_A,
    DEFAULT_MODEL_B,
    load_pair,
    pick_device,
)
from src.tasks.cipher_decode import read_jsonl  # noqa: E402

EXPERIMENTS_DIR = ROOT / "experiments"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--dataset", type=str, default="data/cipher_v3_test.jsonl")
    p.add_argument("--key-direction", type=str, default="decoding",
                   choices=["encoding", "decoding"])
    p.add_argument("--out-name", type=str, default=None)
    p.add_argument("--seed", type=int, default=42, help="Adapter init seed.")
    p.add_argument("--max-new-tokens", type=int, default=500)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.out_name or f"run_phase3_{timestamp}"
    run_dir = EXPERIMENTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    test_path = ROOT / args.dataset
    test = list(read_jsonl(test_path))[: args.n]
    print(
        f"[config] device={device}  n={len(test)}  key_direction={args.key_direction}  "
        f"dataset={args.dataset}  seed={args.seed}  run_dir={run_dir}"
    )

    config = {
        "phase": 3,
        "condition": "random_adapter_baseline",
        "model_a": DEFAULT_MODEL_A,
        "model_b": DEFAULT_MODEL_B,
        "share_model_weights": True,
        "device": device,
        "n_episodes": len(test),
        "key_direction": args.key_direction,
        "test_dataset": str(test_path.relative_to(ROOT)),
        "adapter": "random_orthogonal",
        "adapter_seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "timestamp": timestamp,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    model_a, tok_a, model_b, tok_b = load_pair(
        DEFAULT_MODEL_A, DEFAULT_MODEL_B,
        share_model_weights=True,
    )

    d_a = model_a.config.hidden_size
    d_b = model_b.config.hidden_size
    print(f"[dims] d_A={d_a}  d_B={d_b}")

    torch.manual_seed(args.seed)
    adapter = Adapter(d_a, d_b).to(model_a.device).to(next(model_a.parameters()).dtype)
    init_orthogonal(adapter)
    for p in adapter.parameters():
        p.requires_grad_(False)
    print(f"[adapter] random orthogonal, d_A={d_a} -> d_B={d_b}, frozen")

    log_path = run_dir / "episodes.jsonl"
    accs: list[float] = []
    ems: list[bool] = []
    compls: list[bool] = []
    toks: list[int] = []
    times: list[float] = []

    print(f"\n[run] Logging to {log_path}\n")
    with log_path.open("w") as logf:
        for i, ep in enumerate(test):
            t0 = time.time()
            result = run_episode_adapter(
                model_a, tok_a, model_b, tok_b, adapter, ep,
                key_direction=args.key_direction,
                max_new_tokens=args.max_new_tokens,
            )
            elapsed = time.time() - t0
            result["elapsed_seconds"] = elapsed
            logf.write(json.dumps(result) + "\n")
            logf.flush()

            accs.append(result["char_acc"])
            ems.append(result["exact_match"])
            compls.append(result["answer_tag_found"])
            toks.append(result["tokens_used"])
            times.append(elapsed)

            running_acc = statistics.mean(accs)
            running_em = statistics.mean(ems)
            running_compl = statistics.mean(int(c) for c in compls)
            print(
                f"[{i+1:3d}/{len(test)}] pt={ep['plaintext']!r:>10}  "
                f"pred={result['predicted_plaintext']!r:>10}  "
                f"tag={'Y' if result['answer_tag_found'] else 'N'}  "
                f"acc={result['char_acc']:.2f}  em={int(result['exact_match'])}  "
                f"tok={result['tokens_used']:4d}  norm={result['signal_norm']:.2f}  t={elapsed:5.1f}s  "
                f"running_acc={running_acc:.3f}  em_rate={running_em:.3f}  compl={running_compl:.2f}"
            )

    summary = {
        "n_episodes": len(test),
        "mean_char_accuracy": statistics.mean(accs),
        "stdev_char_accuracy": statistics.stdev(accs) if len(accs) > 1 else 0.0,
        "exact_match_rate": statistics.mean(ems),
        "format_compliance_rate": statistics.mean(int(c) for c in compls),
        "mean_total_tokens": statistics.mean(toks),
        "mean_elapsed_seconds": statistics.mean(times),
        "median_elapsed_seconds": statistics.median(times),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[summary]\n{json.dumps(summary, indent=2)}")
    print(f"\n[done] Outputs in {run_dir}")


if __name__ == "__main__":
    main()
