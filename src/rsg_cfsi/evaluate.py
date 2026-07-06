"""Validation protocol — Section 4 of the RSG-CFSI paper.

Chronological train/test split with a no-reoptimization lock; AUC, Brier
score, calibration bins, lead-time evaluation at 5/10/20-day horizons;
blocked-bootstrap confidence intervals for AUC differences (preferred
under serial dependence). Pure numpy; no sklearn dependency.

Author : Taiko Toeda / MOBIUS LLC
License: AGPL-3.0-or-later
"""
from __future__ import annotations

import numpy as np

HORIZONS = (5, 10, 20)


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank AUC (Mann-Whitney), ties handled by midranks."""
    score = np.asarray(score, float)
    label = np.asarray(label).astype(bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    sorted_scores = score[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def brier(prob: np.ndarray, label: np.ndarray) -> float:
    prob = np.asarray(prob, float)
    label = np.asarray(label, float)
    return float(np.mean((prob - label) ** 2))


def calibration_bins(prob: np.ndarray, label: np.ndarray,
                     n_bins: int = 10) -> list[dict]:
    """Reliability-curve data: mean predicted vs observed rate per bin."""
    prob = np.asarray(prob, float)
    label = np.asarray(label, float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if mask.sum() == 0:
            continue
        rows.append({"bin": [round(float(lo), 2), round(float(hi), 2)],
                     "n": int(mask.sum()),
                     "mean_pred": round(float(prob[mask].mean()), 4),
                     "obs_rate": round(float(label[mask].mean()), 4)})
    return rows


def lead_labels(crisis: np.ndarray, horizon: int) -> np.ndarray:
    """Label 1 if a crisis onset occurs within the next `horizon` steps
    (onset = 0->1 transition). Same-day and later-only reaction is thereby
    separated from advance warning — the Section 4 lead-time concern."""
    crisis = np.asarray(crisis).astype(int)
    onset = np.zeros_like(crisis)
    onset[1:] = (crisis[1:] == 1) & (crisis[:-1] == 0)
    out = np.zeros_like(crisis)
    idx = np.where(onset == 1)[0]
    for i in idx:
        out[max(0, i - horizon):i] = 1
    return out


def blocked_bootstrap_auc_diff(score_a: np.ndarray, score_b: np.ndarray,
                               label: np.ndarray, block: int = 20,
                               n_boot: int = 1000,
                               seed: int = 20260706) -> dict:
    """CI for AUC(a) - AUC(b) via circular block bootstrap (serial
    dependence-aware, per Section 4's preference over DeLong here)."""
    rng = np.random.default_rng(seed)
    T = len(label)
    diffs = []
    for _ in range(n_boot):
        starts = rng.integers(0, T, size=(T // block) + 1)
        idx = np.concatenate([np.arange(s, s + block) % T for s in starts])[:T]
        la = np.asarray(label)[idx]
        if la.sum() == 0 or la.sum() == len(la):
            continue
        diffs.append(auc(np.asarray(score_a)[idx], la)
                     - auc(np.asarray(score_b)[idx], la))
    diffs = np.array(diffs)
    return {"mean_diff": round(float(diffs.mean()), 4),
            "ci95": [round(float(np.percentile(diffs, 2.5)), 4),
                     round(float(np.percentile(diffs, 97.5)), 4)],
            "n_boot_effective": int(len(diffs))}


def evaluate(score: np.ndarray, prob: np.ndarray, crisis: np.ndarray,
             train_end: int) -> dict:
    """Full Section-4 report on the TEST window only."""
    s, p, c = (np.asarray(v)[train_end:] for v in (score, prob, crisis))
    report = {
        "n_test": int(len(s)),
        "auc_same_day": round(auc(s, c.astype(bool)), 4),
        "brier": round(brier(p, c), 4),
        "calibration": calibration_bins(p, c),
        "lead_time": {},
    }
    full_c = np.asarray(crisis)
    for h in HORIZONS:
        ll = lead_labels(full_c, h)[train_end:]
        report["lead_time"][f"{h}d"] = round(auc(s, ll.astype(bool)), 4)
    return report
