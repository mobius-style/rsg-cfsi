"""Model arms — the paper's evidence-ladder comparison set (Sections 5, 8).

Implemented in this MVP release:
  arm0_static     : regularization-only core, adaptive learning OFF
  arm1_core       : full regularization core (Phi + damped momentum + p)
  arm2_no_momentum: ablation — momentum removed (Section 4 ablation row)

Scaffolded but NOT implemented here (honest scope; see limitations.md):
  narrative, network, behavioral-adaptive, and ABM comparison arms
  (ladder layer E), order/path coupling tests (F), and the MDL
  neutral-basis ledger (G). Their registry slots exist so a replication
  package can add them without changing the harness.

Author : Taiko Toeda / MOBIUS LLC
License: AGPL-3.0-or-later
"""
from __future__ import annotations

import numpy as np

from .indicator import CFSIParams, ewi


def arm0_static(x, labels, train_end):
    """Regularization-only baseline: no bounded adaptive learning."""
    return ewi(x, labels, train_end, CFSIParams(),
               use_bounded_learning=False)


def arm1_core(x, labels, train_end):
    """Full reference core (the paper's Section 3 specification)."""
    return ewi(x, labels, train_end, CFSIParams())


def arm2_no_momentum(x, labels, train_end):
    """Ablation arm: damped momentum removed."""
    return ewi(x, labels, train_end, CFSIParams(), use_momentum=False)


ARMS = {
    "arm0_static": arm0_static,
    "arm1_core": arm1_core,
    "arm2_no_momentum": arm2_no_momentum,
    # evidence-ladder E-G slots (not implemented in the MVP):
    "armE_narrative": None,
    "armE_network": None,
    "armE_behavioral_adaptive": None,
    "armE_abm": None,
    "armF_coupling": None,
    "armG_mdl_ledger": None,
}
