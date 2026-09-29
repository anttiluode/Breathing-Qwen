import os

import numpy as np
import pytest

from breathing_qwen.qwen_score import QwenCandidateScorer


@pytest.mark.qwen
def test_local_qwen_scoring_is_deterministic():
    model = os.environ.get("BREATHING_QWEN_MODEL")
    revision = os.environ.get("BREATHING_QWEN_REVISION")
    if not model or not revision:
        pytest.skip("set BREATHING_QWEN_MODEL and BREATHING_QWEN_REVISION for opt-in Qwen test")
    scorer = QwenCandidateScorer.from_pretrained(model, revision=revision)
    a = scorer.score_cue("a red planet with Olympus Mons", ["Mars", "Venus"])
    b = scorer.score_cue("a red planet with Olympus Mons", ["Mars", "Venus"])
    assert np.all(np.isfinite(a))
    assert np.allclose(a, b)
