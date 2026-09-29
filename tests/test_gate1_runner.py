import numpy as np

from breathing_qwen.benchmark import BenchmarkItem
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
