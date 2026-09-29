from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow direct execution from a fresh checkout without requiring installation.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from breathing_qwen.receipts import read_receipt


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Breathing-Qwen JSON receipts without rerunning a model.")
    parser.add_argument("receipts", nargs="+", type=Path)
    args = parser.parse_args()

    for path in args.receipts:
        receipt = read_receipt(path)
        print(f"{path}: {receipt.gate} {receipt.status} | {receipt.model.get('name')} @ {receipt.model.get('revision')}")
        for arm, metrics in receipt.metrics.items():
            if metrics:
                print(
                    f"  {arm:20s} clean={metrics['clean_accuracy']:.3f} "
                    f"corrupt={metrics['corrupt_accuracy']:.3f} "
                    f"rank={metrics['mean_rank']:.3f} margin={metrics['mean_margin']:.4f}"
                )
        if receipt.diagnostics:
            print(f"  diagnostics: {receipt.diagnostics}")
        if receipt.error:
            print(f"  error: {receipt.error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
