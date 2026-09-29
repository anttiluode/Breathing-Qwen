from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class InternalAttentionCompatibilityError(RuntimeError):
    """Raised when a model-specific Gate-3 hook has not been verified."""


@dataclass(frozen=True, order=True)
class AttentionTarget:
    layer: int
    head: int

    def __post_init__(self) -> None:
        if self.layer < 0 or self.head < 0:
            raise ValueError("layer and head must be non-negative")


class InternalAttentionAdapter:
    """Pure intervention boundary for Gate 3; model hooking is intentionally unverified."""

    def __init__(self, selected: set[AttentionTarget] | frozenset[AttentionTarget] = frozenset()):
        self.selected = frozenset(selected)

    def modify_scores(
        self,
        scores: np.ndarray,
        *,
        layer: int,
        head: int,
        beta: float,
        span_bias: np.ndarray | None = None,
    ) -> np.ndarray:
        x = np.asarray(scores, dtype=float)
        if not np.all(np.isfinite(x)):
            raise ValueError("scores must be finite")
        if beta <= 0:
            raise ValueError("beta must be positive")
        target = AttentionTarget(layer, head)
        if target not in self.selected:
            return x.copy()
        out = beta * x
        if span_bias is not None:
            bias = np.asarray(span_bias, dtype=float)
            if bias.shape != x.shape:
                raise ValueError("span_bias shape must match scores")
            if not np.all(np.isfinite(bias)):
                raise ValueError("span_bias must be finite")
            out = out + bias
        return out

    def install(self, model, *, transformers_version: str) -> None:
        raise InternalAttentionCompatibilityError(
            "Gate-3 Qwen attention hooking is not verified for this Transformers/Qwen implementation; "
            f"refusing to patch global model state (transformers={transformers_version})."
        )
