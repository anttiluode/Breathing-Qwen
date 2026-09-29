import numpy as np

from breathing_qwen.toy_memory import ToyGateConfig, make_toy_benchmark, run_toy_gate


def test_toy_benchmark_is_seed_reproducible():
    a = make_toy_benchmark(7, n_items=8, n_memories=8, dim=16)
    b = make_toy_benchmark(7, n_items=8, n_memories=8, dim=16)
    assert np.allclose(a.memories, b.memories)
    for ia, ib in zip(a.items, b.items, strict=True):
        assert ia.target_index == ib.target_index
        assert ia.corrupt_index == ib.corrupt_index
        assert np.allclose(ia.clean_cues, ib.clean_cues)
        assert np.allclose(ia.corrupt_cues, ib.corrupt_cues)


def test_corrupt_item_changes_exactly_one_cue_and_keeps_memories_fixed():
    bench = make_toy_benchmark(9, n_items=12, n_memories=8, dim=16)
    original_memories = bench.memories.copy()
    for item in bench.items:
        changed = np.any(np.abs(item.clean_cues - item.corrupt_cues) > 1e-12, axis=1)
        assert changed.sum() == 1
        assert changed[item.corrupt_index]
        assert item.distractor_index != item.target_index
    assert np.allclose(bench.memories, original_memories)


def test_gate_output_has_required_metrics_and_all_arms():
    result = run_toy_gate(ToyGateConfig(seed=11, n_items=24, n_memories=8, dim=16))
    assert set(result["arms"]) == {"one_shot", "fixed", "breathing", "breathing_residue"}
    for arm in result["arms"].values():
        assert {"clean_accuracy", "corrupt_accuracy", "corruption_penalty", "mean_margin"} <= set(arm)
    assert {"recovery_rate_on_native_misses", "clean_regression_rate", "passed"} <= set(result["gate"])


def test_negative_gate_is_a_scientific_result_not_runtime_failure():
    config = ToyGateConfig(seed=13, n_items=16, n_memories=8, dim=16, delta=1000.0)
    result = run_toy_gate(config)
    assert result["gate"]["passed"] is False
    assert result["gate"]["reason"]
