"""Synthetic stress-market generator for the runnable demo.

Real financial series carry licensing terms; the reference implementation
therefore ships with a synthetic generator so the full pipeline is
runnable out of the box. Regime-switching volatility with planted crisis
episodes; K observable stress inputs (vol proxy, spread proxy, liquidity
proxy, correlation proxy) that load on the latent stress with noise.

Author : Taiko Toeda / MOBIUS LLC
License: AGPL-3.0-or-later
"""
from __future__ import annotations

import numpy as np


def generate(T: int = 3000, K: int = 4, n_crises: int = 5,
             crisis_len: tuple[int, int] = (30, 90),
             seed: int = 20260706) -> dict:
    rng = np.random.default_rng(seed)
    latent = np.zeros(T)
    crisis = np.zeros(T, dtype=int)
    # background AR(1) stress
    for t in range(1, T):
        latent[t] = 0.97 * latent[t - 1] + rng.normal(0, 0.15)
    # planted crisis episodes: stress ramps up BEFORE the labeled window
    starts = np.sort(rng.choice(np.arange(200, T - 200), size=n_crises,
                                replace=False))
    for s in starts:
        length = int(rng.integers(*crisis_len))
        ramp = int(rng.integers(10, 25))
        for i in range(ramp):                      # pre-crisis buildup
            if s - ramp + i >= 0:
                latent[s - ramp + i] += 2.5 * (i / ramp)
        latent[s:s + length] += 2.5
        latent[s + length:s + length + 40] += np.linspace(2.5, 0, 40)[
            : max(0, min(40, T - s - length))]
        crisis[s:s + length] = 1
    loadings = rng.uniform(0.6, 1.2, size=K)
    noise = rng.normal(0, 0.8, size=(T, K))
    x = np.outer(latent, loadings) + noise
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    return {"x": x, "crisis": crisis, "latent": latent,
            "meta": {"T": T, "K": K, "n_crises": n_crises, "seed": seed}}
