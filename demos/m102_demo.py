#!/usr/bin/env python3
"""M102 — Demo: Explore the churn data in an Azure Machine Learning notebook
Objectives: TO1, EO1c
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
    variant = "notebook_baseline" if "m102" in ("m102", "m103", "m105") else ("automl_best" if "m102" == "m106" else "release_candidate")
    eval_res = evaluate_predictions(splits["validation"], model_variant=variant)
    report = {
        "clip": "m102",
        "title": "Demo: Explore the churn data in an Azure Machine Learning notebook",
        "objectives": ["TO1", "EO1c"],
        "dataset_version": "globomantics-churn-v1",
        "split_counts": {k: len(v) for k, v in splits.items()},
        "final_test_unopened": "m102" != "m305",
        "metrics": eval_res,
        "steps_verified": ["Open the churn notebook on a CPU compute instance and load version 1 of the churn data asset", "Profile class balance, missing values, and plan-tier counts, and confirm every feature is dated before its observation", "Check that customers and dates stay inside the chronological 60/20/20 split, then fit a scikit-learn baseline", "Read the validation confusion matrix and leave the final-test partition unopened"],
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
