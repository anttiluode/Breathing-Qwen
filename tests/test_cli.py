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


def test_run_qwen_help_exposes_4bit_loader_flag():
    result = subprocess.run(
        [sys.executable, "scripts/run_qwen.py", "--help"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--load-4bit" in result.stdout


def test_run_requery_help_works_from_fresh_checkout_and_exposes_4bit():
    result = subprocess.run(
        [sys.executable, "scripts/run_requery.py", "--help"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Gate 1B" in result.stdout
    assert "--load-4bit" in result.stdout


def test_run_holdout_help_works_from_fresh_checkout_and_exposes_4bit():
    result = subprocess.run(
        [sys.executable, "scripts/run_holdout.py", "--help"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Gate 1C" in result.stdout
    assert "128-item held-out" in result.stdout
    assert "--load-4bit" in result.stdout


def test_summarize_help_works_from_fresh_checkout_without_install():
    result = subprocess.run(
        [sys.executable, "scripts/summarize_receipts.py", "--help"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Summarize Breathing-Qwen" in result.stdout


def test_run_toy_works_from_fresh_checkout_without_install():
    result = subprocess.run(
        [sys.executable, "scripts/run_toy.py", "--json"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert '"gate_name": "gate0"' in result.stdout
