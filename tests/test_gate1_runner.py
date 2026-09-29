import numpy as np
import pytest

from breathing_qwen.benchmark import BenchmarkItem, V0_BENCHMARK_SHA256
from breathing_qwen.receipts import Gate1Config, run_gate1


class MappingScorer:
    model_name = "fake-model"
    revision = "fake-rev"

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
    return {
        "c1": [1.2, 1.0, 0.0, -0.2],
        "c2": [1.1, 0.9, 0.0, -0.2],
        "c3": [1.0, 0.8, 0.0, -0.2],
        "true": [1.3, 0.7, 0.0, -0.2],
        "false": [0.0, 4.0, 0.0, -0.2],
        "c1\nc2\nc3\ntrue": [3.0, 0.0, -1.0, -2.0],
        "c1\nc2\nc3\nfalse": [0.0, 3.0, -1.0, -2.0],
        "true\nc3\nc2\nc1": [3.0, 0.0, -1.0, -2.0],
        "false\nc3\nc2\nc1": [0.0, 3.0, -1.0, -2.0],
    }


def test_gate1_uses_cached_shared_cue_evidence_and_all_five_arms():
    scorer = MappingScorer(mapping())
    receipt = run_gate1([item()], scorer, Gate1Config(benchmark_hash="abc"))
    assert set(receipt.metrics) == {
        "native_all_cues", "per_cue_one_shot", "fixed", "breathing", "breathing_residue"
    }
    assert len(scorer.calls) == 9
    per_cue_calls = [cue for cue, _ in scorer.calls if cue in {"c1", "c2", "c3", "true", "false"}]
    assert sorted(per_cue_calls) == ["c1", "c2", "c3", "false", "true"]


def test_gate1_reports_recovery_clean_regression_rank_margin_and_trust():
    receipt = run_gate1([item()], MappingScorer(mapping()), Gate1Config(benchmark_hash="abc"))
    assert receipt.metrics["native_all_cues"]["corrupt_accuracy"] == 0.0
    assert receipt.metrics["breathing_residue"]["corrupt_accuracy"] == 1.0
    assert receipt.diagnostics["recovery_rate_on_native_misses"] == 1.0
    assert receipt.diagnostics["clean_regression_rate"] == 0.0
    detail = receipt.items[0]
    assert detail["corrupt_index"] == 3
    final_weights = detail["traces"]["corrupt"]["breathing_residue"][-1]["cue_weights"]
    assert final_weights[3] < min(final_weights[:3])
    assert "mean_rank" in receipt.metrics["breathing_residue"]
    assert "mean_margin" in receipt.metrics["breathing_residue"]


def test_order_control_is_recorded_separately_from_settling():
    receipt = run_gate1([item()], MappingScorer(mapping()), Gate1Config(benchmark_hash="abc"))
    assert receipt.diagnostics["native_order_flip_rate_clean"] == 0.0
    assert receipt.diagnostics["native_order_flip_rate_corrupt"] == 0.0


def test_frozen_gate1_decision_requires_all_three_predeclared_conditions():
    from breathing_qwen.receipts import evaluate_gate1

    metrics = {
        "fixed": {"clean_accuracy": 0.90, "corrupt_accuracy": 0.50},
        "breathing": {"clean_accuracy": 0.91, "corrupt_accuracy": 0.52},
        "breathing_residue": {"clean_accuracy": 0.89, "corrupt_accuracy": 0.63},
    }
    correct = [0] * 10
    native = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    robust = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    decision = evaluate_gate1(metrics, native, robust, correct)
    assert decision["passed"] is True
    assert decision["corrupt_gain"] == pytest.approx(0.11)
    assert decision["clean_loss"] == pytest.approx(0.02)
    assert decision["recoveries"] == 5
    assert decision["new_errors"] == 0
    assert decision["recovery_advantage"] == 5


def test_frozen_gate1_decision_fails_if_recovery_advantage_is_under_four():
    from breathing_qwen.receipts import evaluate_gate1

    metrics = {
        "fixed": {"clean_accuracy": 0.90, "corrupt_accuracy": 0.50},
        "breathing": {"clean_accuracy": 0.91, "corrupt_accuracy": 0.52},
        "breathing_residue": {"clean_accuracy": 0.89, "corrupt_accuracy": 0.63},
    }
    correct = [0] * 6
    native = [1, 1, 1, 0, 0, 0]
    robust = [0, 0, 0, 0, 0, 0]
    decision = evaluate_gate1(metrics, native, robust, correct)
    assert decision["passed"] is False
    assert decision["recovery_advantage"] == 3


def test_partial_smoke_run_never_evaluates_frozen_gate1_decision():
    receipt = run_gate1(
        [item()], MappingScorer(mapping()), Gate1Config(benchmark_hash=V0_BENCHMARK_SHA256)
    )
    assert receipt.diagnostics["gate_decision"]["evaluated"] is False
    assert receipt.diagnostics["gate_decision"]["passed"] is False
    assert "requires 32 completed items; got 1" in receipt.diagnostics["gate_decision"]["reason"]


def test_wrong_32_item_benchmark_hash_cannot_evaluate_official_gate():
    receipt = run_gate1(
        [item()] * 32,
        MappingScorer(mapping()),
        Gate1Config(benchmark_hash="not-the-v0-hash"),
    )
    decision = receipt.diagnostics["gate_decision"]
    assert decision["evaluated"] is False
    assert decision["passed"] is False
    assert "hash" in decision["reason"].lower()
