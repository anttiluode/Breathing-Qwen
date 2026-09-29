from pathlib import Path


def test_readme_reports_gate0_kill_and_does_not_invent_gate1_result():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "Gate 0: FAIL" in text
    assert "+5.47 percentage points" in text
    assert "Gate 1: NOT RUN" in text
    assert "Qwen3-8B" in text


def test_readme_names_external_settling_boundary():
    text = Path("README.md").read_text(encoding="utf-8").lower()
    assert "external settling" in text
    assert "breathing-only" in text
    assert "does not change candidate ordering" in text


def test_readme_states_frozen_gate1_decision_rule_and_full_run_requirement():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "all 32" in text
    assert "+10 percentage points" in text
    assert "4 more recoveries than new errors" in text
