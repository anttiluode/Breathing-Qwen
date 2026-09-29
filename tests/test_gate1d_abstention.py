import importlib.util


def test_gate1d_module_exists_before_any_behavior_is_implemented():
    assert importlib.util.find_spec("breathing_qwen.gate1d") is not None
