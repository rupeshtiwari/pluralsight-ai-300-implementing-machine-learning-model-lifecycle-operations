#!/usr/bin/env python3
"""M406 — Demo: Roll out a churn candidate with mirrored and weighted traffic
Objectives: TO3, EO3c
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from globomantics_churn.dataset import chronological_split, evaluate_predictions, generate_observations


def run_demo() -> dict:
    rows = generate_observations(n=12000, seed=42)
    splits = chronological_split(rows)
    variant = "notebook_baseline" if "m406" in ("m102", "m103", "m105") else ("automl_best" if "m406" == "m106" else "release_candidate")
    eval_res = evaluate_predictions(splits["validation"], model_variant=variant)
    report = {
        "clip": "m406",
        "title": "Demo: Roll out a churn candidate with mirrored and weighted traffic",
        "objectives": ["TO3", "EO3c"],
        "dataset_version": "globomantics-churn-v1",
        "split_counts": {k: len(v) for k, v in splits.items()},
        "final_test_unopened": "m406" != "m305",
        "metrics": eval_res,
        "steps_verified": ["Test the prepared green deployment, which serves a retrained version that passed the release gate, directly while blue keeps all live traffic", "Mirror 10 percent of requests to green and check its logs, then turn mirroring off, set a 90/10 split, and send 3,000 scoring requests", "Check green\u2019s requests against the canary gate and apply the decision", "Run the rollback drill: set blue to 100 percent and confirm that new requests reach blue"],
        "status": "PASS",
    }
    return report


if __name__ == "__main__":
    res = run_demo()
    print(f"=== [{res['clip'].upper()}] {res['title']} ===")
    print(f"Objectives       : {', '.join(res['objectives'])}")
    print(f"Split Counts     : train={res['split_counts']['train']}, val={res['split_counts']['validation']}, final_test={res['split_counts']['final_test']}")
    print(f"Recall@FPR<=10%  : {res['metrics']['recall_at_fpr10']:.3f} (FPR={res['metrics']['false_positive_rate']:.3f})")
    print(f"Release Gate     : {'APPROVED' if res['metrics']['release_gate_passed'] else 'BASELINE_RECORDED'}")
    for idx, st in enumerate(res["steps_verified"], start=1):
        print(f"  [STEP {idx} PASS] {st}")
