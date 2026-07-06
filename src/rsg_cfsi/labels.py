"""Canonical crisis-label construction (monograph Chapter 8.5).

Crisis days = days when ALL input series simultaneously exceed their
q-th percentile (joint exceedance), with percentiles estimated on the
TRAINING window only to avoid look-ahead. The paper's label-dependence
caveat applies verbatim: labels built from the same variables used as
inputs measure internal stress classification, not fully independent
crisis prediction — state this in any release that uses this labeler.

Author : Taiko Toeda / MOBIUS LLC
License: AGPL-3.0-or-later
"""
from __future__ import annotations

import numpy as np


def joint_exceedance_labels(x: np.ndarray, train_end: int,
                            q: float = 90.0) -> np.ndarray:
    """Return 0/1 crisis labels via joint q-th percentile exceedance.

    x: (T, K) input series (same series the indicator consumes).
    Thresholds come from x[:train_end] only.
    """
    x = np.asarray(x, dtype=float)
    thresholds = np.percentile(x[:train_end], q, axis=0)
    return (x > thresholds).all(axis=1).astype(int)
