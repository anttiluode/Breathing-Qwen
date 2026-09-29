from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from breathing_qwen.toy_memory import ToyGateConfig, run_toy_gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Breathing-Qwen CPU Gate 0")
    parser.add_argument("--json", action="store_true", help="emit compact JSON")
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    result = run_toy_gate(ToyGateConfig(seed=args.seed))
    print(json.dumps(result, indent=None if args.json else 2, sort_keys=True))


if __name__ == "__main__":
    main()
