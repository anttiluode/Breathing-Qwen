import json
from pathlib import Path

from breathing_qwen.benchmark import V0_BENCHMARK_SHA256
from breathing_qwen.receipts import Gate1Config, RunReceipt, read_receipt, write_receipt


def minimal_receipt():
    return RunReceipt(
        gate="gate1",
        created_at="2026-09-29T00:00:00Z",
        model={"name": "fake", "revision": "rev"},
        benchmark={"hash": "abc", "items": 1},
        config={"delta": 0.2},
        environment={"python": "x", "numpy": "y"},
        metrics={"native_all_cues": {"clean_accuracy": 1.0}},
        diagnostics={},
        items=[],
        status="complete",
        error=None,
    )


def test_receipt_round_trip(tmp_path: Path):
    path = tmp_path / "r.json"
    write_receipt(minimal_receipt(), path)
    loaded = read_receipt(path)
    assert loaded.to_dict() == minimal_receipt().to_dict()


def test_receipt_json_contains_provenance_and_no_chain_of_thought(tmp_path: Path):
    path = tmp_path / "r.json"
    write_receipt(minimal_receipt(), path)
    raw = json.loads(path.read_text())
    assert {"model", "benchmark", "config", "environment", "metrics", "status"} <= set(raw)
    text = path.read_text().lower()
    assert "chain_of_thought" not in text
    assert "reasoning_text" not in text


def test_gate1_config_freezes_v0_constants():
    cfg = Gate1Config(benchmark_hash="abc")
    assert cfg.delta == 0.2
    assert cfg.weight_floor == 0.05
    assert cfg.schedule.betas == (0.50, 1.75, 0.65, 2.00)
    assert cfg.official_item_count == 32
    assert cfg.official_benchmark_hash == V0_BENCHMARK_SHA256
