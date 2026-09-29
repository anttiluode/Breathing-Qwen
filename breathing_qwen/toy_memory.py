from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .schedules import DEFAULT_SCHEDULE
from .settling import settle

ARMS = ("one_shot", "fixed", "breathing", "breathing_residue")


@dataclass(frozen=True)
class ToyGateConfig:
    seed: int = 1729
    n_items: int = 256
    n_memories: int = 16
    dim: int = 32
    n_cues: int = 3
    pair_separation: float = 0.22
    clean_noise: float = 0.38
    corrupt_noise: float = 0.03
    delta: float = 0.20
    weight_floor: float = 0.05


@dataclass(frozen=True)
class ToyItem:
    target_index: int
    distractor_index: int
    corrupt_index: int
    clean_cues: np.ndarray
    corrupt_cues: np.ndarray


@dataclass(frozen=True)
class ToyBenchmark:
    memories: np.ndarray
    items: tuple[ToyItem, ...]


def _unit(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def _paired_memories(rng: np.random.Generator, n_memories: int, dim: int, separation: float) -> np.ndarray:
    if n_memories < 2 or n_memories % 2:
        raise ValueError("n_memories must be an even integer >= 2")
    rows: list[np.ndarray] = []
    for _ in range(n_memories // 2):
        base = _unit(rng.normal(size=(1, dim)))[0]
        direction = rng.normal(size=dim)
        direction -= base * float(direction @ base)
        direction = _unit(direction[None, :])[0]
        rows.extend((_unit((base + separation * direction)[None, :])[0],
                     _unit((base - separation * direction)[None, :])[0]))
    return np.stack(rows)


def make_toy_benchmark(
    seed: int,
    *,
    n_items: int,
    n_memories: int,
    dim: int,
    n_cues: int = 3,
    pair_separation: float = 0.22,
    clean_noise: float = 0.38,
    corrupt_noise: float = 0.03,
) -> ToyBenchmark:
    if n_items < 1 or n_cues < 2 or dim < 2:
        raise ValueError("n_items >= 1, n_cues >= 2, and dim >= 2 are required")
    rng = np.random.default_rng(seed)
    memories = _paired_memories(rng, n_memories, dim, pair_separation)
    items: list[ToyItem] = []
    for _ in range(n_items):
        target = int(rng.integers(n_memories))
        distractor = target ^ 1
        clean = _unit(memories[target][None, :] + clean_noise * rng.normal(size=(n_cues, dim)))
        corrupt = clean.copy()
        corrupt_index = int(rng.integers(n_cues))
        corrupt[corrupt_index] = _unit(
            memories[distractor][None, :] + corrupt_noise * rng.normal(size=(1, dim))
        )[0]
        items.append(ToyItem(target, distractor, corrupt_index, clean, corrupt))
    return ToyBenchmark(memories=memories, items=tuple(items))


def _evidence(cues: np.ndarray, memories: np.ndarray) -> np.ndarray:
    return cues @ memories.T


def _arm_metrics(bench: ToyBenchmark, mode: str, *, delta: float, weight_floor: float, corrupt: bool) -> tuple[dict[str, float], list[int]]:
    winners: list[int] = []
    margins: list[float] = []
    correct = 0
    for item in bench.items:
        cues = item.corrupt_cues if corrupt else item.clean_cues
        trace = settle(_evidence(cues, bench.memories), DEFAULT_SCHEDULE, mode=mode, delta=delta, weight_floor=weight_floor)
        winners.append(trace.winner_index)
        margins.append(trace.final_margin)
        correct += int(trace.winner_index == item.target_index)
    n = len(bench.items)
    return {"accuracy": correct / n, "mean_margin": float(np.mean(margins))}, winners


def run_toy_gate(config: ToyGateConfig) -> dict[str, Any]:
    bench = make_toy_benchmark(
        config.seed,
        n_items=config.n_items,
        n_memories=config.n_memories,
        dim=config.dim,
        n_cues=config.n_cues,
        pair_separation=config.pair_separation,
        clean_noise=config.clean_noise,
        corrupt_noise=config.corrupt_noise,
    )
    arms: dict[str, dict[str, float]] = {}
    predictions: dict[str, dict[str, list[int]]] = {}
    for mode in ARMS:
        clean, clean_winners = _arm_metrics(bench, mode, delta=config.delta, weight_floor=config.weight_floor, corrupt=False)
        corrupt, corrupt_winners = _arm_metrics(bench, mode, delta=config.delta, weight_floor=config.weight_floor, corrupt=True)
        arms[mode] = {
            "clean_accuracy": clean["accuracy"],
            "corrupt_accuracy": corrupt["accuracy"],
            "corruption_penalty": clean["accuracy"] - corrupt["accuracy"],
            "mean_margin": corrupt["mean_margin"],
        }
        predictions[mode] = {"clean": clean_winners, "corrupt": corrupt_winners}

    native = predictions["one_shot"]
    robust = predictions["breathing_residue"]
    native_misses = [i for i, item in enumerate(bench.items) if native["corrupt"][i] != item.target_index]
    recovered = sum(robust["corrupt"][i] == bench.items[i].target_index for i in native_misses)
    native_clean_correct = [i for i, item in enumerate(bench.items) if native["clean"][i] == item.target_index]
    regressions = sum(robust["clean"][i] != bench.items[i].target_index for i in native_clean_correct)
    recovery_rate = recovered / len(native_misses) if native_misses else 0.0
    clean_regression_rate = regressions / len(native_clean_correct) if native_clean_correct else 0.0

    controls = ("one_shot", "fixed", "breathing")
    best_corrupt_control = max(arms[name]["corrupt_accuracy"] for name in controls)
    best_clean_control = max(arms[name]["clean_accuracy"] for name in controls)
    robust_arm = arms["breathing_residue"]
    corrupt_gain = robust_arm["corrupt_accuracy"] - best_corrupt_control
    clean_loss = best_clean_control - robust_arm["clean_accuracy"]
    passed = corrupt_gain >= 0.10 and clean_loss <= 0.02
    reason = (
        f"corrupt_gain={corrupt_gain:.4f} (need >=0.1000); "
        f"clean_loss={clean_loss:.4f} (need <=0.0200)"
    )
    return {
        "gate_name": "gate0",
        "config": asdict(config),
        "arms": arms,
        "gate": {
            "passed": bool(passed),
            "reason": reason,
            "recovery_rate_on_native_misses": recovery_rate,
            "clean_regression_rate": clean_regression_rate,
            "best_corrupt_control": best_corrupt_control,
            "best_clean_control": best_clean_control,
        },
    }
