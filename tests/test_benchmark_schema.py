from pathlib import Path

from breathing_qwen.benchmark import load_benchmark


BENCHMARK = Path("benchmarks/v0.jsonl")


def test_v0_has_32_unique_items():
    items = load_benchmark(BENCHMARK)
    assert len(items) == 32
    assert len({item.id for item in items}) == 32


def test_each_item_is_a_single_cue_corruption_pair():
    for item in load_benchmark(BENCHMARK):
        assert item.candidates.count(item.answer) == 1
        assert len(item.clean_cues) == len(item.corrupt_cues)
        changed = [i for i, (a, b) in enumerate(zip(item.clean_cues, item.corrupt_cues, strict=True)) if a != b]
        assert changed == [item.corrupt_index]
        assert len(item.candidates) >= 4


def test_inference_view_is_blind_to_labels():
    item = load_benchmark(BENCHMARK)[0]
    view = item.inference_view(corrupt=True)
    assert set(view) == {"id", "cues", "candidates"}
    assert "answer" not in view
    assert "corrupt_index" not in view
    assert view["cues"] == list(item.corrupt_cues)


def test_clean_and_corrupt_views_keep_candidate_order_fixed():
    for item in load_benchmark(BENCHMARK):
        assert item.inference_view(False)["candidates"] == item.inference_view(True)["candidates"]
