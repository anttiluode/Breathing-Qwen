from __future__ import annotations

import subprocess
import sys


def test_run_qwen_help_works_from_fresh_checkout_without_install():
    result = subprocess.run(
        [sys.executable, "scripts/run_qwen.py", "--help"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Qwen3-8B Gate 1" in result.stdout


def test_summarize_help_works_from_fresh_checkout_without_install():
    result = subprocess.run(
        [sys.executable, "scripts/summarize_receipts.py", "--help"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Summarize Breathing-Qwen" in result.stdout
