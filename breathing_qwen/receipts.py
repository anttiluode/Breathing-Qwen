from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .benchmark import BenchmarkItem
from .schedules import DEFAULT_SCHEDULE, BreathingSchedule
from .settling import SettlingTrace, settle

ARMS = ("native_all_cues", "per_cue_one_shot", "fixed", "breathing", "breathing_residue")


@dataclass(frozen=True)
class Gate1Config:
    benchmark_hash: str
    schedule: BreathingSchedule = DEFAULT_SCHEDULE
    delta: float = 0.20
    weight_floor: float = 0.05
    include_order_control: bool = True


@dataclass
class RunReceipt:
    gate: str
    created_at: str
    model: dict[str, Any]
    benchmark: dict[str, Any]
    config: dict[str, Any]
    environment: dict[str, Any]
    metrics: dict[str, Any]
    diagnostics: dict[str, Any]
    items: list[dict[str, Any]]
    status: str
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RunReceipt":
        return cls(**raw)


def write_receipt(receipt: RunReceipt, path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_receipt(path: Path | str) -> RunReceipt:
    return RunReceipt.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, timeout=2
        ).strip()
    except Exception:
        return "unknown"


def _softmax(scores: np.ndarray) -> np.ndarray:
    x = np.asarray(scores, dtype=float)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _rank_margin(prob: np.ndarray, correct: int) -> tuple[int, float]:
    correct_p = float(prob[correct])
    rank = 1 + int(np.sum(prob > correct_p))
    if len(prob) == 1:
        margin = correct_p
    else:
        other = float(np.max(np.delete(prob, correct)))
        margin = correct_p - other
    return rank, margin


def _cycle_dict(trace: SettlingTrace) -> list[dict[str, Any]]:
    return [
        {
            "cycle": cycle.cycle,
            "phase": cycle.phase,
            "beta": cycle.beta,
            "aggregate_scores": cycle.aggregate_scores.tolist(),
            "probabilities": cycle.probabilities.tolist(),
            "entropy": cycle.entropy,
            "residuals": cycle.residuals.tolist(),
            "cue_weights": cycle.cue_weights.tolist(),
        }
        for cycle in trace.cycles
    ]


def _empty_stats() -> dict[str, dict[str, list[float]]]:
    return {
        arm: {
            "clean_correct": [], "clean_rank": [], "clean_margin": [],
            "corrupt_correct": [], "corrupt_rank": [], "corrupt_margin": [],
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


def run_gate1(
    items: Sequence[BenchmarkItem],
    scorer,
    config: Gate1Config,
) -> RunReceipt:
    stats = _empty_stats()
    item_rows: list[dict[str, Any]] = []
    cache: dict[tuple[tuple[str, ...], str], np.ndarray] = {}
    native_clean_winners: list[int] = []
    native_corrupt_winners: list[int] = []
    robust_clean_winners: list[int] = []
    robust_corrupt_winners: list[int] = []
    correct_indices: list[int] = []
    clean_flips = 0
    corrupt_flips = 0
    corrupt_final_weights: list[float] = []
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
            clean_evidence = np.vstack([score(cue, candidates) for cue in item.clean_cues])
            corrupt_evidence = np.vstack([score(cue, candidates) for cue in item.corrupt_cues])

            traces: dict[str, dict[str, list[dict[str, Any]]]] = {"clean": {}, "corrupt": {}}
            probs_by_condition: dict[str, dict[str, np.ndarray]] = {"clean": {}, "corrupt": {}}

            for condition, evidence in (("clean", clean_evidence), ("corrupt", corrupt_evidence)):
                cue_texts = item.clean_cues if condition == "clean" else item.corrupt_cues
                native_scores = score("\n".join(cue_texts), candidates)
                probs_by_condition[condition]["native_all_cues"] = _softmax(native_scores)

                for arm, mode in (
                    ("per_cue_one_shot", "one_shot"),
                    ("fixed", "fixed"),
                    ("breathing", "breathing"),
                    ("breathing_residue", "breathing_residue"),
                ):
                    trace = settle(
                        evidence,
                        config.schedule,
                        mode=mode,
                        delta=config.delta,
                        weight_floor=config.weight_floor,
                    )
                    probs_by_condition[condition][arm] = trace.final_probabilities
                    traces[condition][arm] = _cycle_dict(trace)

                if config.include_order_control:
                    reversed_scores = score("\n".join(reversed(cue_texts)), candidates)
                    flipped = int(np.argmax(reversed_scores)) != int(np.argmax(native_scores))
                    if condition == "clean":
                        clean_flips += int(flipped)
                    else:
                        corrupt_flips += int(flipped)

            predictions: dict[str, dict[str, int]] = {"clean": {}, "corrupt": {}}
            for condition in ("clean", "corrupt"):
                for arm in ARMS:
                    prob = probs_by_condition[condition][arm]
                    winner = int(np.argmax(prob))
                    predictions[condition][arm] = winner
                    rank, margin = _rank_margin(prob, correct)
                    stats[arm][f"{condition}_correct"].append(float(winner == correct))
                    stats[arm][f"{condition}_rank"].append(float(rank))
                    stats[arm][f"{condition}_margin"].append(margin)

            native_clean_winners.append(predictions["clean"]["native_all_cues"])
            native_corrupt_winners.append(predictions["corrupt"]["native_all_cues"])
            robust_clean_winners.append(predictions["clean"]["breathing_residue"])
            robust_corrupt_winners.append(predictions["corrupt"]["breathing_residue"])
            corrupt_weights = traces["corrupt"]["breathing_residue"][-1]["cue_weights"]
            corrupt_final_weights.append(float(corrupt_weights[item.corrupt_index]))

            item_rows.append(
                {
                    "id": item.id,
                    "answer": item.answer,
                    "correct_index": correct,
                    "corrupt_index": item.corrupt_index,
                    "candidates": list(candidates),
                    "predictions": predictions,
                    "traces": traces,
                }
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    n = len(item_rows)
    native_misses = [i for i in range(n) if native_corrupt_winners[i] != correct_indices[i]]
    native_clean_correct = [i for i in range(n) if native_clean_winners[i] == correct_indices[i]]
    recovery = (
        sum(robust_corrupt_winners[i] == correct_indices[i] for i in native_misses) / len(native_misses)
        if native_misses else 0.0
    )
    regression = (
        sum(robust_clean_winners[i] != correct_indices[i] for i in native_clean_correct) / len(native_clean_correct)
        if native_clean_correct else 0.0
    )
    diagnostics = {
        "recovery_rate_on_native_misses": float(recovery),
        "clean_regression_rate": float(regression),
        "native_order_flip_rate_clean": clean_flips / n if n and config.include_order_control else 0.0,
        "native_order_flip_rate_corrupt": corrupt_flips / n if n and config.include_order_control else 0.0,
        "mean_final_weight_known_corrupt_cue": float(np.mean(corrupt_final_weights)) if corrupt_final_weights else None,
        "cached_score_vectors": len(cache),
    }
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return RunReceipt(
        gate="gate1",
        created_at=created_at,
        model={"name": getattr(scorer, "model_name", "unknown"), "revision": getattr(scorer, "revision", None)},
        benchmark={"hash": config.benchmark_hash, "items": len(items), "completed_items": n},
        config={
            "delta": config.delta,
            "weight_floor": config.weight_floor,
            "schedule": {"betas": list(config.schedule.betas), "phases": list(config.schedule.phases)},
            "include_order_control": config.include_order_control,
        },
        environment={
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "git_commit": _git_sha(),
        },
        metrics=_finalize_metrics(stats),
        diagnostics=diagnostics,
        items=item_rows,
        status="complete" if error is None else "failed",
        error=error,
    )
