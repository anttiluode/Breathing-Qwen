from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np

from .benchmark import BenchmarkItem, V0_BENCHMARK_SHA256
from .receipts import RunReceipt
from .schedules import DEFAULT_SCHEDULE, BreathingSchedule
from .settling import settle

ARMS = (
    "native_all_cues",
    "random_remove",
    "residue_guided",
    "oracle_remove",
    "exhaustive_leave_one_out",
)


@dataclass(frozen=True)
class Gate1BConfig:
    benchmark_hash: str
    schedule: BreathingSchedule = DEFAULT_SCHEDULE
    delta: float = 0.20
    weight_floor: float = 0.05
    official_item_count: int = 32
    official_benchmark_hash: str = V0_BENCHMARK_SHA256


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, timeout=2
        ).strip()
    except Exception:
        return "unknown"


def _softmax(scores: np.ndarray) -> np.ndarray:
    x = np.asarray(scores, dtype=float)
    if x.ndim != 1 or not x.size or not np.all(np.isfinite(x)):
        raise ValueError("scores must be a finite nonempty vector")
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _rank_margin(prob: np.ndarray, correct: int) -> tuple[int, float]:
    correct_p = float(prob[correct])
    rank = 1 + int(np.sum(prob > correct_p))
    other = float(np.max(np.delete(prob, correct))) if len(prob) > 1 else 0.0
    return rank, correct_p - other


def deterministic_random_index(item_id: str, cue_count: int) -> int:
    if cue_count < 1:
        raise ValueError("cue_count must be positive")
    digest = hashlib.sha256(f"gate1b-v0:{item_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % cue_count


def ensemble_leave_one_out(score_vectors: Sequence[np.ndarray]) -> np.ndarray:
    if not score_vectors:
        raise ValueError("score_vectors must be nonempty")
    probs = [_softmax(np.asarray(scores, dtype=float)) for scores in score_vectors]
    width = len(probs[0])
    if any(len(p) != width for p in probs):
        raise ValueError("all score vectors must have equal length")
    return np.mean(np.vstack(probs), axis=0)


def _empty_stats() -> dict[str, dict[str, list[float]]]:
    return {
        arm: {
            "clean_correct": [],
            "clean_rank": [],
            "clean_margin": [],
            "corrupt_correct": [],
            "corrupt_rank": [],
            "corrupt_margin": [],
        }
        for arm in ARMS
    }


def _finalize_metrics(stats: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for arm, s in stats.items():
        if not s["clean_correct"]:
            metrics[arm] = {}
            continue
        clean_acc = float(np.mean(s["clean_correct"]))
        corrupt_acc = float(np.mean(s["corrupt_correct"]))
        metrics[arm] = {
            "clean_accuracy": clean_acc,
            "corrupt_accuracy": corrupt_acc,
            "corruption_penalty": clean_acc - corrupt_acc,
            "mean_rank": float(np.mean(s["corrupt_rank"])),
            "mean_margin": float(np.mean(s["corrupt_margin"])),
        }
    return metrics


def evaluate_gate1b(
    metrics: dict[str, dict[str, float]],
    native_corrupt_winners: Sequence[int],
    guided_corrupt_winners: Sequence[int],
    correct_indices: Sequence[int],
) -> dict[str, Any]:
    required = ("native_all_cues", "random_remove", "residue_guided")
    if any(not metrics.get(name) for name in required):
        return {"evaluated": False, "passed": False, "reason": "insufficient completed metrics"}
    if not (
        len(native_corrupt_winners) == len(guided_corrupt_winners) == len(correct_indices)
    ):
        raise ValueError("winner and correct-index vectors must have equal length")

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
        "min_gain_over_native": 0.0625,
        "min_gain_over_random": 0.0625,
        "max_clean_loss": 0.03125,
        "min_recovery_advantage": 2,
    }
    passed = (
        gain_over_native >= thresholds["min_gain_over_native"]
        and gain_over_random >= thresholds["min_gain_over_random"]
        and clean_loss <= thresholds["max_clean_loss"]
        and recovery_advantage >= thresholds["min_recovery_advantage"]
    )
    return {
        "evaluated": True,
        "passed": bool(passed),
        "gain_over_native": float(gain_over_native),
        "gain_over_random": float(gain_over_random),
        "clean_loss": float(clean_loss),
        "recoveries": int(recoveries),
        "new_errors": int(new_errors),
        "recovery_advantage": recovery_advantage,
        "thresholds": thresholds,
        "reason": (
            f"gain_over_native={gain_over_native:.4f} (need >=0.0625); "
            f"gain_over_random={gain_over_random:.4f} (need >=0.0625); "
            f"clean_loss={clean_loss:.4f} (need <=0.03125); "
            f"recovery_advantage={recovery_advantage} (need >=2)"
        ),
    }


def run_gate1b(
    items: Sequence[BenchmarkItem],
    scorer,
    config: Gate1BConfig,
) -> RunReceipt:
    stats = _empty_stats()
    item_rows: list[dict[str, Any]] = []
    cache: dict[tuple[tuple[str, ...], str], np.ndarray] = {}
    native_corrupt_winners: list[int] = []
    guided_corrupt_winners: list[int] = []
    correct_indices: list[int] = []
    locator_hits: list[float] = []
    error: str | None = None

    def score(text: str, candidates: tuple[str, ...]) -> np.ndarray:
        key = (candidates, text)
        if key not in cache:
            value = np.asarray(scorer.score_cue(text, candidates), dtype=float)
            if value.shape != (len(candidates),) or not np.all(np.isfinite(value)):
                raise ValueError("scorer returned invalid candidate score vector")
            cache[key] = value
        return cache[key].copy()

    try:
        for item in items:
            candidates = item.candidates
            correct = candidates.index(item.answer)
            correct_indices.append(correct)
            predictions: dict[str, dict[str, int]] = {"clean": {}, "corrupt": {}}
            selected: dict[str, int] = {}
            final_weights: dict[str, list[float]] = {}

            for condition, cues in (("clean", item.clean_cues), ("corrupt", item.corrupt_cues)):
                native_prob = _softmax(score("\n".join(cues), candidates))
                evidence = np.vstack([score(cue, candidates) for cue in cues])
                trace = settle(
                    evidence,
                    config.schedule,
                    mode="breathing_residue",
                    delta=config.delta,
                    weight_floor=config.weight_floor,
                )
                weights = trace.cycles[-1].cue_weights
                guided_index = int(np.argmin(weights))
                random_index = deterministic_random_index(item.id, len(cues))
                oracle_index = item.corrupt_index
                selected[condition] = guided_index
                final_weights[condition] = weights.tolist()

                loo_scores: list[np.ndarray] = []
                for remove_index in range(len(cues)):
                    kept = tuple(cue for idx, cue in enumerate(cues) if idx != remove_index)
                    loo_scores.append(score("\n".join(kept), candidates))

                probs = {
                    "native_all_cues": native_prob,
                    "random_remove": _softmax(loo_scores[random_index]),
                    "residue_guided": _softmax(loo_scores[guided_index]),
                    "oracle_remove": _softmax(loo_scores[oracle_index]),
                    "exhaustive_leave_one_out": ensemble_leave_one_out(loo_scores),
                }
                for arm, prob in probs.items():
                    winner = int(np.argmax(prob))
                    predictions[condition][arm] = winner
                    rank, margin = _rank_margin(prob, correct)
                    stats[arm][f"{condition}_correct"].append(float(winner == correct))
                    stats[arm][f"{condition}_rank"].append(float(rank))
                    stats[arm][f"{condition}_margin"].append(float(margin))

            native_corrupt_winners.append(predictions["corrupt"]["native_all_cues"])
            guided_corrupt_winners.append(predictions["corrupt"]["residue_guided"])
            locator_correct = selected["corrupt"] == item.corrupt_index
            locator_hits.append(float(locator_correct))
            item_rows.append(
                {
                    "id": item.id,
                    "answer": item.answer,
                    "correct_index": correct,
                    "corrupt_index": item.corrupt_index,
                    "candidates": list(candidates),
                    "selected_remove_index": selected,
                    "final_cue_weights": final_weights,
                    "random_remove_index": deterministic_random_index(item.id, len(item.clean_cues)),
                    "locator_correct": bool(locator_correct),
                    "predictions": predictions,
                }
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    n = len(item_rows)
    metrics = _finalize_metrics(stats)
    diagnostics: dict[str, Any] = {
        "locator_accuracy_corrupt": float(np.mean(locator_hits)) if locator_hits else None,
        "cached_score_vectors": len(cache),
        "development_set": True,
    }
    if (
        error is None
        and n == len(items) == config.official_item_count
        and config.benchmark_hash == config.official_benchmark_hash
    ):
        diagnostics["gate_decision"] = evaluate_gate1b(
            metrics, native_corrupt_winners, guided_corrupt_winners, correct_indices
        )
    else:
        if config.benchmark_hash != config.official_benchmark_hash:
            reason = (
                f"benchmark hash is not frozen v0: got {config.benchmark_hash}, "
                f"expected {config.official_benchmark_hash}"
            )
        else:
            reason = f"frozen Gate 1B requires {config.official_item_count} completed items; got {n}"
        diagnostics["gate_decision"] = {"evaluated": False, "passed": False, "reason": reason}

    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return RunReceipt(
        gate="gate1b",
        created_at=created_at,
        model={
            "name": getattr(scorer, "model_name", "unknown"),
            "revision": getattr(scorer, "revision", None),
            "inference_mode": getattr(scorer, "inference_mode", "unknown"),
        },
        benchmark={"hash": config.benchmark_hash, "items": len(items), "completed_items": n},
        config={
            "delta": config.delta,
            "weight_floor": config.weight_floor,
            "schedule": {"betas": list(config.schedule.betas), "phases": list(config.schedule.phases)},
            "official_item_count": config.official_item_count,
            "official_benchmark_hash": config.official_benchmark_hash,
            "development_set": True,
        },
        environment={
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "git_commit": _git_sha(),
        },
        metrics=metrics,
        diagnostics=diagnostics,
        items=item_rows,
        status="complete" if error is None else "failed",
        error=error,
    )
