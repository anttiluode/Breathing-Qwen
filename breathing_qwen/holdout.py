from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .benchmark import BenchmarkItem, V1_HOLDOUT_SHA256
from .requery import Gate1BConfig, run_gate1b
from .receipts import RunReceipt
from .schedules import DEFAULT_SCHEDULE, BreathingSchedule


@dataclass(frozen=True)
class Gate1CConfig:
    benchmark_hash: str
    schedule: BreathingSchedule = DEFAULT_SCHEDULE
    delta: float = 0.20
    weight_floor: float = 0.05
    official_item_count: int = 128
    official_benchmark_hash: str = V1_HOLDOUT_SHA256
    official_model_name: str = "Qwen/Qwen3-8B"
    official_revision: str = "b968826d9c46dd6066d109eabc6255188de91218"
    official_inference_mode: str = "nf4"


def evaluate_gate1c(
    metrics: dict[str, dict[str, float]],
    native_corrupt_winners: Sequence[int],
    guided_corrupt_winners: Sequence[int],
    correct_indices: Sequence[int],
    *,
    locator_accuracy: float,
) -> dict[str, Any]:
    required = ("native_all_cues", "random_remove", "residue_guided")
    if any(not metrics.get(name) for name in required):
        return {"evaluated": False, "passed": False, "reason": "insufficient completed metrics"}
    if not (
        len(native_corrupt_winners) == len(guided_corrupt_winners) == len(correct_indices)
    ):
        raise ValueError("winner and correct-index vectors must have equal length")
    if not np.isfinite(locator_accuracy) or not 0.0 <= locator_accuracy <= 1.0:
        raise ValueError("locator_accuracy must be finite and in [0, 1]")

    native = metrics["native_all_cues"]
    random = metrics["random_remove"]
    guided = metrics["residue_guided"]
    gain_over_native = guided["corrupt_accuracy"] - native["corrupt_accuracy"]
    gain_over_random = guided["corrupt_accuracy"] - random["corrupt_accuracy"]
    clean_loss = native["clean_accuracy"] - guided["clean_accuracy"]
    recoveries = sum(
        n != c and g == c
        for n, g, c in zip(native_corrupt_winners, guided_corrupt_winners, correct_indices, strict=True)
    )
    new_errors = sum(
        n == c and g != c
        for n, g, c in zip(native_corrupt_winners, guided_corrupt_winners, correct_indices, strict=True)
    )
    recovery_advantage = int(recoveries - new_errors)
    thresholds = {
        "min_locator_accuracy": 0.75,
        "min_gain_over_native": 0.0625,
        "min_gain_over_random": 0.0625,
        "max_clean_loss": 0.03125,
        "min_recovery_advantage": 8,
    }
    passed = (
        locator_accuracy >= thresholds["min_locator_accuracy"]
        and gain_over_native >= thresholds["min_gain_over_native"]
        and gain_over_random >= thresholds["min_gain_over_random"]
        and clean_loss <= thresholds["max_clean_loss"]
        and recovery_advantage >= thresholds["min_recovery_advantage"]
    )
    return {
        "evaluated": True,
        "passed": bool(passed),
        "locator_accuracy": float(locator_accuracy),
        "gain_over_native": float(gain_over_native),
        "gain_over_random": float(gain_over_random),
        "clean_loss": float(clean_loss),
        "recoveries": int(recoveries),
        "new_errors": int(new_errors),
        "recovery_advantage": recovery_advantage,
        "thresholds": thresholds,
        "reason": (
            f"locator_accuracy={locator_accuracy:.4f} (need >=0.7500); "
            f"gain_over_native={gain_over_native:.4f} (need >=0.0625); "
            f"gain_over_random={gain_over_random:.4f} (need >=0.0625); "
            f"clean_loss={clean_loss:.4f} (need <=0.03125); "
            f"recovery_advantage={recovery_advantage} (need >=8)"
        ),
    }


def run_gate1c(
    items: Sequence[BenchmarkItem],
    scorer,
    config: Gate1CConfig,
) -> RunReceipt:
    """Run the unchanged Gate-1B mechanism on the frozen, unseen Gate-1C holdout."""
    engine_config = Gate1BConfig(
        benchmark_hash=config.benchmark_hash,
        schedule=config.schedule,
        delta=config.delta,
        weight_floor=config.weight_floor,
        official_item_count=config.official_item_count,
        official_benchmark_hash=config.official_benchmark_hash,
    )
    receipt = run_gate1b(items, scorer, engine_config)
    receipt.gate = "gate1c"
    receipt.config["development_set"] = False
    receipt.config["official_model_name"] = config.official_model_name
    receipt.config["official_revision"] = config.official_revision
    receipt.config["official_inference_mode"] = config.official_inference_mode
    receipt.diagnostics["development_set"] = False

    n = int(receipt.benchmark["completed_items"])
    provenance_ok = (
        receipt.model.get("name") == config.official_model_name
        and receipt.model.get("revision") == config.official_revision
        and receipt.model.get("inference_mode") == config.official_inference_mode
    )
    if (
        receipt.error is None
        and n == len(items) == config.official_item_count
        and config.benchmark_hash == config.official_benchmark_hash
        and provenance_ok
    ):
        native = [row["predictions"]["corrupt"]["native_all_cues"] for row in receipt.items]
        guided = [row["predictions"]["corrupt"]["residue_guided"] for row in receipt.items]
        correct = [row["correct_index"] for row in receipt.items]
        locator_accuracy = float(receipt.diagnostics["locator_accuracy_corrupt"])
        receipt.diagnostics["gate_decision"] = evaluate_gate1c(
            receipt.metrics,
            native,
            guided,
            correct,
            locator_accuracy=locator_accuracy,
        )
    else:
        if config.benchmark_hash != config.official_benchmark_hash:
            reason = (
                f"benchmark hash is not frozen Gate 1C holdout: got {config.benchmark_hash}, "
                f"expected {config.official_benchmark_hash}"
            )
        elif not provenance_ok and n == config.official_item_count:
            reason = (
                "model provenance does not match frozen Gate 1C: "
                f"got {receipt.model.get('name')} @ {receipt.model.get('revision')} "
                f"[{receipt.model.get('inference_mode')}], expected {config.official_model_name} @ "
                f"{config.official_revision} [{config.official_inference_mode}]"
            )
        else:
            reason = f"frozen Gate 1C requires {config.official_item_count} completed items; got {n}"
        receipt.diagnostics["gate_decision"] = {
            "evaluated": False,
            "passed": False,
            "reason": reason,
        }
    return receipt
