import numpy as np
import pytest

from breathing_qwen.internal_attention import (
    AttentionTarget,
    InternalAttentionAdapter,
    InternalAttentionCompatibilityError,
)


def test_unselected_head_is_exact_noop():
    adapter = InternalAttentionAdapter({AttentionTarget(5, 3)})
    scores = np.array([[1.0, 2.0, 3.0]])
    bias = np.array([[0.5, -0.5, 0.0]])
    out = adapter.modify_scores(scores, layer=4, head=3, beta=2.0, span_bias=bias)
    assert np.array_equal(out, scores)
    assert out is not scores


def test_selected_head_applies_beta_and_span_bias_only_locally():
    adapter = InternalAttentionAdapter({AttentionTarget(5, 3)})
    scores = np.array([[1.0, 2.0, 3.0]])
    bias = np.array([[0.5, -0.5, 0.0]])
    out = adapter.modify_scores(scores, layer=5, head=3, beta=2.0, span_bias=bias)
    assert np.allclose(out, [[2.5, 3.5, 6.0]])


def test_adapter_rejects_bad_bias_shape():
    adapter = InternalAttentionAdapter({AttentionTarget(5, 3)})
    with pytest.raises(ValueError, match="shape"):
        adapter.modify_scores(
            np.array([[1.0, 2.0, 3.0]]),
            layer=5,
            head=3,
            beta=1.0,
            span_bias=np.array([1.0, 2.0]),
        )


def test_install_refuses_unverified_qwen_hook_instead_of_global_patch():
    adapter = InternalAttentionAdapter({AttentionTarget(5, 3)})
    with pytest.raises(InternalAttentionCompatibilityError, match="not verified"):
        adapter.install(object(), transformers_version="unknown")
