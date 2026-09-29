from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

# Allow direct execution from a fresh checkout without requiring installation.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from breathing_qwen.benchmark import V0_BENCHMARK_SHA256, load_benchmark
from breathing_qwen.qwen_score import QwenCandidateScorer
from breathing_qwen.receipts import Gate1Config, run_gate1, write_receipt


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_max_memory(raw: str | None) -> dict | None:
    if raw is None:
        return None
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--max-memory-json must decode to an object")
    result = {}
    for key, value in parsed.items():
        normalized = int(key) if str(key).isdigit() else key
        result[normalized] = value
    return result


def default_output() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("receipts") / f"gate1_{stamp}.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run frozen Qwen3-8B Gate 1 on the Breathing-Qwen v0 benchmark."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", required=True, help="exact model revision/commit; required for receipts")
    parser.add_argument("--benchmark", type=Path, default=Path("benchmarks/v0.jsonl"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="first N items for runtime smoke test; 0 means all")
    parser.add_argument("--max-memory-json", help='Transformers max_memory JSON, e.g. {"0":"6GiB","cpu":"6GiB"}')
    parser.add_argument("--offload-folder", type=str)
    parser.add_argument("--candidate-batch-size", type=int, default=1)
    parser.add_argument("--no-order-control", action="store_true")
    parser.add_argument(
        "--allow-benchmark-mismatch",
        action="store_true",
        help="allow a benchmark whose bytes differ from frozen v0 (receipt still records its hash)",
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be >= 0")
    if args.candidate_batch_size < 1:
        parser.error("--candidate-batch-size must be >= 1")

    benchmark_hash = file_sha256(args.benchmark)
    if (
        args.benchmark.name == "v0.jsonl"
        and benchmark_hash != V0_BENCHMARK_SHA256
        and not args.allow_benchmark_mismatch
    ):
        parser.error(
            f"v0 benchmark hash mismatch: got {benchmark_hash}, expected {V0_BENCHMARK_SHA256}; "
            "do not silently edit v0"
        )

    items = load_benchmark(args.benchmark)
    if args.limit:
        items = items[: args.limit]
    if not items:
        parser.error("benchmark contains no selected items")

    try:
        max_memory = parse_max_memory(args.max_memory_json)
    except (ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    scorer = QwenCandidateScorer.from_pretrained(
        args.model,
        revision=args.revision,
        max_memory=max_memory,
        offload_folder=args.offload_folder,
        candidate_batch_size=args.candidate_batch_size,
    )
    config = Gate1Config(
        benchmark_hash=benchmark_hash,
        include_order_control=not args.no_order_control,
    )
    receipt = run_gate1(items, scorer, config)
    output = args.output or default_output()
    write_receipt(receipt, output)

    print(f"receipt: {output}")
    print(f"status: {receipt.status}")
    print(f"model: {receipt.model['name']} @ {receipt.model['revision']}")
    print(
        f"benchmark: {benchmark_hash} "
        f"({receipt.benchmark['completed_items']}/{receipt.benchmark['items']} completed)"
    )
    for arm, metrics in receipt.metrics.items():
        if metrics:
            print(
                f"{arm}: clean={metrics['clean_accuracy']:.3f} "
                f"corrupt={metrics['corrupt_accuracy']:.3f} "
                f"rank={metrics['mean_rank']:.3f} margin={metrics['mean_margin']:.4f}"
            )
    decision = receipt.diagnostics.get("gate_decision", {})
    print(f"gate decision: {decision}")
    if receipt.error:
        print(f"error: {receipt.error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
