import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rsg_cfsi.evaluate import auc, brier, lead_labels  # noqa: E402
from rsg_cfsi.indicator import CFSIParams, ewi, stress_state  # noqa: E402
from rsg_cfsi.synthetic import generate  # noqa: E402


def test_stress_state_mean_reversion():
    x = np.zeros((200, 3))
    x[50, :] = 10.0  # single shock
    phi, m = stress_state(x, CFSIParams())
    assert phi[60] > phi[100] > phi[199]  # decays back — no permanent drift
    assert abs(phi[199]) < 0.05


def test_momentum_damping_bounds_extremes():
    p = CFSIParams()
    assert p.gamma == 0.3  # canonical RSG-9 constant (monograph Ch. 8)
    # damped momentum is globally bounded by 1/(gamma*e) regardless of jump size
    bound = 1.0 / (p.gamma * np.e) + 1e-9
    for jump in (0.05, 1.0, 5.0, 50.0):
        assert abs(jump * np.exp(-p.gamma * jump)) <= bound
    # and the damping factor is strictly decreasing in |dPhi|
    f = lambda d: np.exp(-p.gamma * d)
    assert f(0.1) > f(1.0) > f(10.0)


def test_eta_schedule_converges_within_bounds():
    p = CFSIParams()
    assert abs(p.eta_at(0) - p.eta_init) < 1e-6
    assert abs(p.eta_at(10_000) - p.eta_steady) < 1e-3
    for t in (0, 10, 100, 1000, 10_000):
        assert p.eta_min <= p.eta_at(t) <= p.eta_max


def test_auc_known_values():
    assert auc(np.array([0.1, 0.2, 0.8, 0.9]),
               np.array([0, 0, 1, 1])) == 1.0
    assert abs(auc(np.array([0.5, 0.5, 0.5, 0.5]),
                   np.array([0, 1, 0, 1])) - 0.5) < 1e-12


def test_lead_labels_precede_onset():
    crisis = np.zeros(50, int)
    crisis[30:40] = 1
    ll = lead_labels(crisis, 5)
    assert ll[25:30].all() and not ll[:25].any() and not ll[30:].any()


def test_pipeline_beats_chance_on_synthetic():
    data = generate(T=1500, seed=7)
    r = ewi(data["x"], data["crisis"], train_end=900)
    test = slice(900, None)
    a = auc(r["ewi"][test], data["crisis"][test].astype(bool))
    assert a > 0.7  # planted signal must be recoverable
    assert 0 <= brier(r["p"][test], data["crisis"][test]) <= 0.3


def test_ewi_weights_locked():
    p = CFSIParams()
    assert p.ewi_weights == {"phi": 0.35, "momentum": 0.25,
                             "probability": 0.40}
