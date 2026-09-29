import numpy as np
import pytest

from breathing_qwen.robust import huber_weights


def test_huber_weights_keep_small_residuals_at_full_trust():
    w = huber_weights(np.array([0.0, 0.5, 1.5]), delta=1.5, floor=0.05)
    assert np.allclose(w, [1.0, 1.0, 1.0])


def test_huber_weights_decrease_monotonically_after_delta():
    w = huber_weights(np.array([1.0, 2.0, 4.0]), delta=1.5, floor=0.05)
    assert w[0] >= w[1] >= w[2]
    assert w[0] == pytest.approx(1.0)
    assert w[2] < 1.0


def test_huber_weights_never_delete_a_cue():
    w = huber_weights(np.array([1e9]), delta=1.5, floor=0.07)
    assert w[0] == pytest.approx(0.07)


def test_huber_weights_reject_nonfinite_residuals():
    with pytest.raises(ValueError, match="finite"):
        huber_weights(np.array([0.0, np.nan]), delta=1.5, floor=0.05)
