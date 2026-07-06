"""RSG-CFSI core indicator — reference implementation.

Implements the regularized stress-state core of the paper
"RSG-CFSI: A Regularized Financial Stress Indicator with Out-of-Sample
Stability" (MOBIUS Technical Companion Papers, No. 1), Section 3:

    Phi_t = alpha * Phi_{t-1} + (1 - alpha) * (w_t . x_t)      (mean reversion)
    eta_min <= eta_t <= eta_max                                 (bounded learning)
    m_t   = dPhi_t * exp(-gamma * |dPhi_t|)                     (damped momentum)
    EWI_t = 0.35 * norm(Phi_t) + 0.25 * norm(m_t) + 0.40 * norm(p_t)

Phi_t is a scalar stress-state proxy; "curvature" names the degree of
stress deformation in the selected market state space, not a literal
tensor. The EWI weights are the paper's REFERENCE weights: releases must
report sensitivity to alternatives and must not tune them on test data.
Normalization constants are estimated on the training window only.

Author : Taiko Toeda / MOBIUS LLC
License: AGPL-3.0-or-later
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

REFERENCE_WEIGHTS = {"phi": 0.35, "momentum": 0.25, "probability": 0.40}


@dataclass
class CFSIParams:
    alpha: float = 0.94          # mean-reversion memory of the stress state
    gamma: float = 2.0           # momentum damping strength
    eta: float = 0.05            # adaptive learning rate for input weights
    eta_min: float = 0.01        # bounded-learning floor
    eta_max: float = 0.20        # bounded-learning ceiling
    ewi_weights: dict = field(default_factory=lambda: dict(REFERENCE_WEIGHTS))

    def __post_init__(self) -> None:
        if not (0.0 <= self.alpha < 1.0):
            raise ValueError("alpha must be in [0, 1)")
        if not (self.eta_min <= self.eta <= self.eta_max):
            raise ValueError("eta must satisfy eta_min <= eta <= eta_max")
        if abs(sum(self.ewi_weights.values()) - 1.0) > 1e-9:
            raise ValueError("EWI weights must sum to 1")


@dataclass
class TrainScaling:
    """Normalization constants — estimated on TRAINING data only (Section 4:
    no re-optimization on the test window)."""
    phi_mean: float
    phi_std: float
    m_mean: float
    m_std: float

    def norm_phi(self, v: np.ndarray) -> np.ndarray:
        return _squash((v - self.phi_mean) / self.phi_std)

    def norm_m(self, v: np.ndarray) -> np.ndarray:
        return _squash((v - self.m_mean) / self.m_std)


def _squash(z: np.ndarray) -> np.ndarray:
    """Map a z-scored series into [0, 1] via the logistic function."""
    return 1.0 / (1.0 + np.exp(-z))


def stress_state(x: np.ndarray, params: CFSIParams,
                 adaptive: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Run the regularized stress-state recursion over inputs x (T x K).

    Input weights w_t start uniform and, when `adaptive` is on, move by
    bounded gradient steps toward inputs that co-move with the current
    state innovation — the paper's bounded adaptive learning. Weights are
    kept non-negative and renormalized each step so the state stays a
    weighted average of stress inputs. Returns (Phi, m): the stress state
    and the damped momentum, both length T.
    """
    x = np.asarray(x, dtype=float)
    T, K = x.shape
    w = np.full(K, 1.0 / K)
    phi = np.zeros(T)
    m = np.zeros(T)
    prev_phi = 0.0
    for t in range(T):
        signal = float(w @ x[t])
        phi[t] = params.alpha * prev_phi + (1.0 - params.alpha) * signal
        d_phi = phi[t] - prev_phi
        m[t] = d_phi * np.exp(-params.gamma * abs(d_phi))
        if adaptive and t > 0:
            eta_t = float(np.clip(params.eta, params.eta_min, params.eta_max))
            w = w + eta_t * d_phi * (x[t] - signal)
            w = np.clip(w, 0.0, None)
            s = w.sum()
            w = np.full(K, 1.0 / K) if s <= 0 else w / s
        prev_phi = phi[t]
    return phi, m


def crisis_probability(phi: np.ndarray, labels: np.ndarray,
                       train_end: int) -> tuple[np.ndarray, tuple[float, float]]:
    """Logistic crisis-probability component p_t, fit on the training window
    only (simple 1-D logistic regression of label on Phi, gradient fit)."""
    zt = (phi - phi[:train_end].mean()) / (phi[:train_end].std() + 1e-12)
    a, b = 0.0, 1.0
    y = labels[:train_end].astype(float)
    z = zt[:train_end]
    for _ in range(500):
        p = 1.0 / (1.0 + np.exp(-(a + b * z)))
        ga = (p - y).mean()
        gb = ((p - y) * z).mean()
        a -= 0.5 * ga
        b -= 0.5 * gb
    return 1.0 / (1.0 + np.exp(-(a + b * zt))), (a, b)


def ewi(x: np.ndarray, labels: np.ndarray, train_end: int,
        params: CFSIParams | None = None,
        use_momentum: bool = True,
        use_bounded_learning: bool = True) -> dict:
    """Full reference pipeline: state -> momentum -> probability -> EWI.

    The `use_momentum` / `use_bounded_learning` switches exist for the
    Section 4 ablation row ("remove momentum, bounded learning, observer
    variables"). Returns a dict with phi, m, p, ewi and the train-window
    scaling actually used.
    """
    params = params or CFSIParams()
    phi, m = stress_state(x, params, adaptive=use_bounded_learning)
    scale = TrainScaling(
        phi_mean=float(phi[:train_end].mean()),
        phi_std=float(phi[:train_end].std() + 1e-12),
        m_mean=float(m[:train_end].mean()),
        m_std=float(m[:train_end].std() + 1e-12),
    )
    p, coef = crisis_probability(phi, labels, train_end)
    wts = params.ewi_weights
    m_term = scale.norm_m(m) if use_momentum else 0.5 * np.ones_like(m)
    index = (wts["phi"] * scale.norm_phi(phi)
             + wts["momentum"] * m_term
             + wts["probability"] * p)
    return {"phi": phi, "m": m, "p": p, "ewi": index,
            "scaling": scale, "logit_coef": coef}
