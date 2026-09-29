from __future__ import annotations

import numpy as np


def huber_weights(residuals: np.ndarray, *, delta: float, floor: float) -> np.ndarray:
    r = np.asarray(residuals, dtype=float)
    if r.ndim != 1:
        raise ValueError("residuals must be a 1D array")
    if not np.all(np.isfinite(r)):
        raise ValueError("residuals must be finite")
    if delta <= 0:
        raise ValueError("delta must be positive")
    if not 0 < floor <= 1:
        raise ValueError("floor must be in (0, 1]")
    a = np.abs(r)
    weights = np.ones_like(a)
    mask = a > delta
    weights[mask] = delta / a[mask]
    return np.clip(weights, floor, 1.0)
