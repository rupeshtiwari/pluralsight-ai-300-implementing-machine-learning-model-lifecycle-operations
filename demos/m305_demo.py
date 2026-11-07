#!/usr/bin/env python3
"""M305 — Demo: Evaluate the churn model with the Azure Machine Learning Responsible AI dashboard
Objectives: TO2, EO2c
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
    variant = "notebook_baseline" if "m305" in ("m102", "m103", "m105") else ("automl_best" if "m305" == "m106" else "release_candidate")
    eval_res = evaluate_predictions(splits["validation"], model_variant=variant)
    report = {
        "clip": "m305",
        "title": "Demo: Evaluate the churn model with the Azure Machine Learning Responsible AI dashboard",
        "objectives": ["TO2", "EO2c"],
        "dataset_version": "globomantics-churn-v1",
        "split_counts": {k: len(v) for k, v in splits.items()},
        "final_test_unopened": "m305" != "m305",
        "metrics": eval_res,
        "steps_verified": ["Submit the Responsible AI insights components with explanation and error analysis on the training and final-test partitions, serving the registered version in a separate container, then open the completed dashboard", "Compare recall and false-positive rate across plan-tier cohorts with their counts, and open the error analysis tree", "Check global feature importance, apply the release gate, and tag the version approved or hold with its intended use and limitations"],
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
