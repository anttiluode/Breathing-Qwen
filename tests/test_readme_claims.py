from pathlib import Path


def test_readme_reports_gate0_gate1q4_gate1b_and_unrun_gate1c():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "Gate 0" in text and "FAIL" in text
    assert "+5.47 pp" in text
    assert "Gate 1-Q4" in text and "FAIL" in text
    assert "Gate 1B" in text and "PASS" in text
    assert "93.75%" in text
    assert "6 recoveries" in text and "0 new errors" in text
    assert "Gate 1C" in text and "NOT RUN" in text
    assert "c0d45ec644f4755f8bfb879457a9b1ae1311c20b4697d0aead9dfb83510dafed" in text


def test_readme_names_external_settling_boundary():
    text = Path("README.md").read_text(encoding="utf-8").lower()
    assert "external beta" in text
    assert "fixed candidate score vector" in text
    assert "internal attention-temperature" in text


def test_readme_states_frozen_gate1c_rule_and_holdout_boundary():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "128" in text
    assert "75%" in text
    assert "8/128 = 6.25 pp" in text
    assert "4/128 = 3.125 pp" in text
    assert "recoveries minus new errors" in text
    assert "held-out" in text.lower()
    assert "b968826d9c46dd6066d109eabc6255188de91218" in text
    assert "NF4" in text
