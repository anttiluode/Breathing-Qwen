from pathlib import Path


def test_readme_reports_gate0_and_gate1q4_failures_without_inventing_gate1b():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "Gate 0" in text and "FAIL" in text
    assert "+5.47 pp" in text
    assert "Gate 1-Q4" in text and "FAIL" in text
    assert "Gate 1B" in text and "NOT RUN" in text
    assert "90.6%" in text
    assert "65.6%" in text
    assert "3 recoveries and 6 new errors" in text


def test_readme_names_external_settling_boundary():
    text = Path("README.md").read_text(encoding="utf-8").lower()
    assert "external beta" in text
    assert "fixed candidate score vector" in text
    assert "internal attention-temperature" in text


def test_readme_states_frozen_gate1b_decision_rule_and_development_boundary():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "2/32 = 6.25 pp" in text
    assert "1/32 = 3.125 pp" in text
    assert "2 more recoveries than new errors" in text
    assert "development-set evidence only" in text
