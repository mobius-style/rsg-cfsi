# RSG-CFSI — reference implementation

**Regularized financial stress indicator with an out-of-sample discipline.**
This is the separately published reference implementation designated by the
paper *"RSG-CFSI: A Regularized Financial Stress Indicator with Out-of-Sample
Stability — Baseline Separation, Theory Discrimination, and Complexity
Accounting"* (MOBIUS Technical Companion Papers, No. 1; Zenodo DOI on the
paper record). Code is **AGPL-3.0-or-later**; the paper text is all rights
reserved and is not included here.

## What is implemented (MVP scope)

The paper's Section-3 core with the monograph's canonical constants
(*Reflective Economics & Financial Engineering*, Chapter 8, RSG-1/5/9):

```
Phi_t = alpha * Phi_{t-1} + (1 - alpha) * (w_t · x_t)   # mean reversion (RSG-1)
eta_t : eta_init → eta_steady, bounded [eta_min, eta_max] # bounded adaptive learning (RSG-5)
m_t   = ΔPhi_t * exp(-γ|ΔPhi_t|),  γ = 0.3               # damped momentum (RSG-9)
EWI_t = 0.35·Φ̃_t + 0.25·m̃_t + 0.40·p̃_t                  # Φ̃, m̃: train-window min-max
```

The canonical crisis labeler (Chapter 8.5: joint 90th-percentile
exceedance of all inputs, thresholds from the training window) ships as
`labels.joint_exceedance_labels`. The monograph's real-data benchmark
(VIX + TED + BAA-AAA, 2004–2014 train / 2015–2021 test) is the
replication target: EWI alone test AUC ≈ 0.826 (stable vs VIX's 0.90 →
0.81 degradation), combined ≈ 0.967; lead-time AUC 0.791 / 0.762 / 0.746
at 5/10/20 days. This repository ships no market data; reproduce those
numbers with your own licensed series via the config templates.

plus the Section-4 validation protocol: chronological train/test lock (no
re-optimization on test), rank AUC, Brier score, calibration bins,
**lead-time AUC at 5/10/20-step horizons** (advance warning scored separately
from same-day classification), ablation arms, and blocked-bootstrap
confidence intervals for AUC differences. Normalization constants and the
crisis-probability logit are estimated on the training window only. EWI
weights are the paper's reference weights and are locked in code.

## Quickstart

```bash
python3 -m pytest tests/          # 6 tests
python3 examples/run_demo.py      # synthetic end-to-end run
```

The demo generates regime-switching synthetic data with planted crisis
episodes (real market series carry licensing terms; none ship here), runs
the implemented arms, and writes the two artifacts the paper's
replication-package table names: `out/train_test_lock.json` and
`out/results_report.md`.

Illustrative demo output (synthetic; validates the *pipeline*, not
real-market performance): removing damped momentum degrades 20-day
lead-time AUC from ≈0.73 to ≈0.65 while leaving same-day AUC almost
unchanged — the momentum term earns its place on exactly the axis
(advance warning) the paper argues matters.

## What is deliberately NOT here

See [limitations.md](limitations.md). In short: evidence-ladder layers E–G
(narrative/network/behavioral/ABM comparison arms, order-path coupling
tests, the MDL neutral-basis ledger) are registry-scaffolded but not
implemented; no real market data; no performance claims. The synthetic
demo cannot demonstrate the value of bounded adaptive learning by
construction (its factor loadings are static) — that arm exists for real
-data replications.

## Layout

```
src/rsg_cfsi/indicator.py   Section-3 core (state, momentum, probability, EWI)
src/rsg_cfsi/evaluate.py    Section-4 protocol (AUC/Brier/calibration/lead-time/bootstrap)
src/rsg_cfsi/arms.py        arm registry (implemented + scaffolded slots)
src/rsg_cfsi/synthetic.py   synthetic generator for the runnable demo
examples/run_demo.py        end-to-end demo -> out/results_report.md
configs/                    train_test_lock / data_manifest templates
tests/                      unit + pipeline tests
```

## Citation

See [CITATION.cff](CITATION.cff). Please cite the paper for the method and
this repository for the implementation.

## License

AGPL-3.0-or-later (see [LICENSE](LICENSE)). © 2026 Taiko Toeda / MOBIUS LLC.
Part of the [MOBIUS project](https://github.com/mobius-style).
