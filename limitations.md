# Limitations (this repository)

This file mirrors the paper's replication-package requirement that
limitations ship as a first-class artifact.

1. **MVP scope.** Implemented: the Section-3 indicator core and the
   Section-4 validation protocol with ablation arms. Not implemented:
   evidence-ladder layers E–G — narrative, network, behavioral-adaptive,
   and ABM comparison arms; order/path coupling-restriction tests; the
   MDL neutral-basis description-length ledger. Registry slots exist in
   `arms.py` so replications can add them without harness changes.
2. **No real market data, no performance claims.** The shipped demo runs
   on synthetic data with planted crises. Its numbers validate the
   pipeline mechanics only. The paper's illustrative 2004–2021 results
   are a starting benchmark to be reproduced independently, not evidence
   this repository claims.
3. **Synthetic-data blind spot.** The generator uses static factor
   loadings, so bounded adaptive learning cannot show value here by
   construction (arm1 ≈ arm0 on the demo). Distinguishing those arms
   requires regime-shifting real data.
4. **Label dependence.** As the paper states: if crisis labels are
   constructed from the same variables used as inputs, results measure
   internal stress classification, not independent crisis prediction.
   The demo's labels are planted and therefore independent of the inputs
   only up to the generator's design.
5. **Single-author implementation.** The code was produced inside the
   same project that wrote the paper (with an AI drafting partner) and
   has not been independently reviewed. The tests are necessary, not
   sufficient.
