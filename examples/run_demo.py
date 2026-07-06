#!/usr/bin/env python3
"""End-to-end demo: synthetic data -> arms -> Section-4 report.

Produces results_report.md + train_test_lock.json in ./out — the same
artifact names the paper's replication-package table specifies.

    python3 examples/run_demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rsg_cfsi.arms import ARMS  # noqa: E402
from rsg_cfsi.evaluate import blocked_bootstrap_auc_diff, evaluate  # noqa: E402
from rsg_cfsi.synthetic import generate  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"


def main() -> int:
    OUT.mkdir(exist_ok=True)
    data = generate()
    x, crisis = data["x"], data["crisis"]
    train_end = int(len(crisis) * 0.6)

    lock = {"split": "chronological", "train_end_index": train_end,
            "reoptimization_on_test": False,
            "labels": "planted synthetic crises (see synthetic.py)",
            "data": data["meta"], "ewi_weights": "reference (0.35/0.25/0.40)"}
    (OUT / "train_test_lock.json").write_text(
        json.dumps(lock, indent=2), encoding="utf-8")

    reports, scores = {}, {}
    for name, fn in ARMS.items():
        if fn is None:
            continue
        r = fn(x, crisis, train_end)
        reports[name] = evaluate(r["ewi"], r["p"], crisis, train_end)
        scores[name] = r["ewi"]
        print(f"{name}: same-day AUC={reports[name]['auc_same_day']} "
              f"Brier={reports[name]['brier']} "
              f"lead={reports[name]['lead_time']}")

    diff = blocked_bootstrap_auc_diff(
        scores["arm1_core"][train_end:], scores["arm0_static"][train_end:],
        crisis[train_end:])
    print(f"AUC(core) - AUC(static) = {diff['mean_diff']} "
          f"CI95={diff['ci95']}")

    lines = ["# RSG-CFSI demo results (synthetic data)", "",
             "Synthetic-data demonstration of the Section-4 protocol. These",
             "numbers validate the PIPELINE, not the indicator's real-market",
             "performance — see limitations.md.", ""]
    for name, rep in reports.items():
        lines += [f"## {name}", "",
                  f"- same-day AUC: {rep['auc_same_day']}",
                  f"- Brier: {rep['brier']}",
                  f"- lead-time AUC: {rep['lead_time']}", ""]
    lines += ["## Arm comparison (blocked bootstrap)", "",
              f"AUC(arm1_core) − AUC(arm0_static) = {diff['mean_diff']}, "
              f"95% CI {diff['ci95']} "
              f"(n_boot={diff['n_boot_effective']})", ""]
    (OUT / "results_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}/results_report.md and train_test_lock.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
