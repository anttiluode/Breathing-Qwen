from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .robust import huber_weights
from .schedules import BreathingSchedule

_MODES = {"one_shot", "fixed", "breathing", "breathing_residue"}


@dataclass(frozen=True)
class SettlingCycle:
    cycle: int
    phase: str
    beta: float
    aggregate_scores: np.ndarray
    probabilities: np.ndarray
    entropy: float
    residuals: np.ndarray
    cue_weights: np.ndarray


@dataclass(frozen=True)
class SettlingTrace:
    mode: str
    cycles: tuple[SettlingCycle, ...]
    final_probabilities: np.ndarray
    winner_index: int
    final_margin: float


def _normalize_rows(evidence: np.ndarray) -> np.ndarray:
    centered = evidence - evidence.mean(axis=1, keepdims=True)
    row_norms = np.linalg.norm(centered, axis=1)
    positive = row_norms[row_norms > 1e-12]
    scale = float(np.median(positive)) if positive.size else 1.0
    return centered / max(scale, 1e-12)


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores)
    exp = np.exp(shifted)
    return exp / exp.sum()


def _entropy(probabilities: np.ndarray) -> float:
    p = np.clip(probabilities, 1e-15, 1.0)
    return float(-(p * np.log(p)).sum())


def _margin(probabilities: np.ndarray) -> float:
    if len(probabilities) < 2:
        return float(probabilities[0])
    top = np.partition(probabilities, -2)[-2:]
    return float(top.max() - top.min())


def settle(
    evidence: np.ndarray,
    schedule: BreathingSchedule,
    *,
    mode: str,
    delta: float,
    weight_floor: float,
) -> SettlingTrace:
    if mode not in _MODES:
        raise ValueError(f"unknown mode: {mode}")
    x = np.asarray(evidence, dtype=float)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("evidence must have shape (n_cues, n_candidates)")
    if not np.all(np.isfinite(x)):
        raise ValueError("evidence must be finite")
    if delta <= 0:
        raise ValueError("delta must be positive")
    if not 0 < weight_floor <= 1:
        raise ValueError("weight_floor must be in (0, 1]")

    x = _normalize_rows(x)
    weights = np.ones(x.shape[0], dtype=float)
    cycles: list[SettlingCycle] = []

    if mode == "one_shot":
        cycle_defs = ((1.0, "one_shot"),)
    elif mode == "fixed":
        cycle_defs = tuple((1.0, phase) for phase in schedule.phases)
    else:
        cycle_defs = tuple(zip(schedule.betas, schedule.phases, strict=True))

    for idx, (beta, phase) in enumerate(cycle_defs):
        aggregate = np.average(x, axis=0, weights=weights)
        probabilities = _softmax(float(beta) * aggregate)
        residuals = np.linalg.norm(x - aggregate[None, :], axis=1)
        cycles.append(
            SettlingCycle(
                cycle=idx,
                phase=phase,
                beta=float(beta),
                aggregate_scores=aggregate.copy(),
                probabilities=probabilities.copy(),
                entropy=_entropy(probabilities),
                residuals=residuals.copy(),
                cue_weights=weights.copy(),
            )
        )
        if mode == "breathing_residue":
            weights = huber_weights(residuals, delta=delta, floor=weight_floor)

    final = cycles[-1].probabilities
    return SettlingTrace(
        mode=mode,
        cycles=tuple(cycles),
        final_probabilities=final.copy(),
        winner_index=int(np.argmax(final)),
        final_margin=_margin(final),
    )
