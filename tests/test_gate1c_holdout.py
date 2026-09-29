import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from breathing_qwen.benchmark import V1_HOLDOUT_SHA256, load_benchmark
from breathing_qwen.holdout import Gate1CConfig, evaluate_gate1c, run_gate1c


HOLDOUT = Path("benchmarks/v1_holdout.jsonl")
EXPECTED_SHA = "c0d45ec644f4755f8bfb879457a9b1ae1311c20b4697d0aead9dfb83510dafed"


class MappingScorer:
    model_name = "fake-model"
    revision = "fake-rev"
    inference_mode = "native"

    def __init__(self, mapping):
        self.mapping = mapping

    def score_cue(self, cue, candidates):
        return np.asarray(self.mapping[cue], dtype=float)


def _tiny_item_and_mapping():
    from breathing_qwen.benchmark import BenchmarkItem

    item = BenchmarkItem(
        id="holdout-x",
        answer="A",
        clean_cues=("c1", "c2", "c3", "true"),
        corrupt_cues=("c1", "c2", "c3", "false"),
        corrupt_index=3,
        candidates=("A", "B", "C", "D"),
    )
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
    return item, m


def test_holdout_bytes_are_frozen_and_balanced():
    raw = HOLDOUT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA == V1_HOLDOUT_SHA256
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    assert len(rows) == 128
    assert len({row["id"] for row in rows}) == 128
    assert Counter(row["domain"] for row in rows) == {domain: 8 for domain in {
        "neuro", "cellbio", "chem", "physics", "astro", "geo", "history", "literature",
        "music", "art", "computing", "math", "instrument", "animal", "plant", "object",
    }}
    assert Counter(row["corrupt_index"] for row in rows) == {0: 32, 1: 32, 2: 32, 3: 32}


def test_holdout_targets_do_not_reuse_v0_targets():
    v0_answers = {item.answer for item in load_benchmark("benchmarks/v0.jsonl")}
    holdout = load_benchmark(HOLDOUT)
    assert len(holdout) == 128
    assert v0_answers.isdisjoint({item.answer for item in holdout})
    assert all(len(item.clean_cues) == len(item.corrupt_cues) == 4 for item in holdout)
    assert all(len(item.candidates) == 5 for item in holdout)


def test_gate1c_frozen_rule_includes_locator_and_scaled_thresholds():
    metrics = {
        "native_all_cues": {"clean_accuracy": 0.90, "corrupt_accuracy": 0.70},
        "random_remove": {"clean_accuracy": 0.88, "corrupt_accuracy": 0.68},
        "residue_guided": {"clean_accuracy": 0.88, "corrupt_accuracy": 0.78},
    }
    correct = [0] * 16
    native = [1] * 8 + [0] * 8
    guided = [0] * 16
    decision = evaluate_gate1c(metrics, native, guided, correct, locator_accuracy=0.80)
    assert decision["passed"] is True
    assert decision["gain_over_native"] == pytest.approx(0.08)
    assert decision["gain_over_random"] == pytest.approx(0.10)
    assert decision["clean_loss"] == pytest.approx(0.02)
    assert decision["recovery_advantage"] == 8
    assert decision["locator_accuracy"] == pytest.approx(0.80)
    assert decision["thresholds"] == {
        "min_locator_accuracy": 0.75,
        "min_gain_over_native": 0.0625,
        "min_gain_over_random": 0.0625,
        "max_clean_loss": 0.03125,
        "min_recovery_advantage": 8,
    }


def test_gate1c_fails_when_locator_is_below_75_percent():
    metrics = {
        "native_all_cues": {"clean_accuracy": 0.90, "corrupt_accuracy": 0.60},
        "random_remove": {"clean_accuracy": 0.88, "corrupt_accuracy": 0.60},
        "residue_guided": {"clean_accuracy": 0.90, "corrupt_accuracy": 0.90},
    }
    correct = [0] * 16
    native = [1] * 8 + [0] * 8
    guided = [0] * 16
    decision = evaluate_gate1c(metrics, native, guided, correct, locator_accuracy=0.74)
    assert decision["passed"] is False
    assert decision["locator_accuracy"] == pytest.approx(0.74)


def test_gate1c_receipt_is_holdout_not_development_set():
    item, mapping = _tiny_item_and_mapping()
    receipt = run_gate1c(
        [item],
        MappingScorer(mapping),
        Gate1CConfig(benchmark_hash=V1_HOLDOUT_SHA256),
    )
    assert receipt.gate == "gate1c"
    assert receipt.config["development_set"] is False
    assert receipt.diagnostics["development_set"] is False
    assert receipt.diagnostics["gate_decision"]["evaluated"] is False
    assert "requires 128 completed items; got 1" in receipt.diagnostics["gate_decision"]["reason"]
