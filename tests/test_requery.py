import numpy as np
import pytest

from breathing_qwen.benchmark import BenchmarkItem, V0_BENCHMARK_SHA256
from breathing_qwen.requery import (
    Gate1BConfig,
    deterministic_random_index,
    ensemble_leave_one_out,
    evaluate_gate1b,
    run_gate1b,
)


class MappingScorer:
    model_name = "fake-model"
    revision = "fake-rev"
    inference_mode = "native"

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def score_cue(self, cue, candidates):
        self.calls.append((cue, tuple(candidates)))
        return np.asarray(self.mapping[cue], dtype=float)


def item():
    return BenchmarkItem(
        id="x",
        answer="A",
        clean_cues=("c1", "c2", "c3", "true"),
        corrupt_cues=("c1", "c2", "c3", "false"),
        corrupt_index=3,
        candidates=("A", "B", "C", "D"),
    )


def mapping():
    m = {
        "c1": [1.2, 1.0, 0.0, -0.2],
        "c2": [1.1, 0.9, 0.0, -0.2],
        "c3": [1.0, 0.8, 0.0, -0.2],
        "true": [1.3, 0.7, 0.0, -0.2],
        "false": [0.0, 4.0, 0.0, -0.2],
        "c1\nc2\nc3\ntrue": [3.0, 0.0, -1.0, -2.0],
        "c1\nc2\nc3\nfalse": [0.0, 3.0, -1.0, -2.0],
    }
    for cues in (
        ("c2", "c3", "true"), ("c1", "c3", "true"),
        ("c1", "c2", "true"), ("c1", "c2", "c3"),
        ("c2", "c3", "false"), ("c1", "c3", "false"),
        ("c1", "c2", "false"),
    ):
        key = "\n".join(cues)
        m[key] = [3.0, 0.0, -1.0, -2.0] if "false" not in cues else [0.0, 2.0, -1.0, -2.0]
    return m


def test_deterministic_random_index_is_frozen_and_platform_independent():
    assert deterministic_random_index("x", 4) == 1
    assert deterministic_random_index("x", 4) == deterministic_random_index("x", 4)


def test_ensemble_leave_one_out_averages_probabilities_not_raw_scores():
    scores = [np.array([10.0, 9.0]), np.array([-100.0, -1.0])]
    probs = ensemble_leave_one_out(scores)
    expected = np.mean([
        np.exp([0.0, -1.0]) / np.exp([0.0, -1.0]).sum(),
        np.exp([-99.0, 0.0]) / np.exp([-99.0, 0.0]).sum(),
    ], axis=0)
    assert np.allclose(probs, expected)


def test_gate1b_uses_residue_only_to_choose_removal_then_requeries_jointly():
    scorer = MappingScorer(mapping())
    receipt = run_gate1b([item()], scorer, Gate1BConfig(benchmark_hash="abc"))
    assert set(receipt.metrics) == {
        "native_all_cues", "random_remove", "residue_guided", "oracle_remove", "exhaustive_leave_one_out"
    }
    detail = receipt.items[0]
    assert detail["selected_remove_index"]["corrupt"] == 3
    assert detail["predictions"]["corrupt"]["native_all_cues"] == 1
    assert detail["predictions"]["corrupt"]["residue_guided"] == 0
    assert detail["predictions"]["corrupt"]["oracle_remove"] == 0
    assert detail["locator_correct"] is True
    assert receipt.model["inference_mode"] == "native"


def test_gate1b_frozen_decision_requires_all_four_conditions():
    metrics = {
        "native_all_cues": {"clean_accuracy": 0.90, "corrupt_accuracy": 0.70},
        "random_remove": {"clean_accuracy": 0.88, "corrupt_accuracy": 0.68},
        "residue_guided": {"clean_accuracy": 0.88, "corrupt_accuracy": 0.78},
    }
    correct = [0] * 10
    native = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    guided = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    decision = evaluate_gate1b(metrics, native, guided, correct)
    assert decision["passed"] is True
    assert decision["gain_over_native"] == pytest.approx(0.08)
    assert decision["gain_over_random"] == pytest.approx(0.10)
    assert decision["clean_loss"] == pytest.approx(0.02)
    assert decision["recovery_advantage"] == 3
    assert decision["thresholds"] == {
        "min_gain_over_native": 0.0625,
        "min_gain_over_random": 0.0625,
        "max_clean_loss": 0.03125,
        "min_recovery_advantage": 2,
    }


def test_gate1b_partial_run_never_evaluates_frozen_decision():
    receipt = run_gate1b(
        [item()], MappingScorer(mapping()), Gate1BConfig(benchmark_hash=V0_BENCHMARK_SHA256)
    )
    decision = receipt.diagnostics["gate_decision"]
    assert decision["evaluated"] is False
    assert decision["passed"] is False
    assert "requires 32 completed items; got 1" in decision["reason"]
