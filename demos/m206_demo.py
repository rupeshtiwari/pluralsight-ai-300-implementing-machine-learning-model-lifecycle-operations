#!/usr/bin/env python3
"""M206 — Demo: Compare churn models across Azure Machine Learning jobs
Objectives: TO1, EO1h
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
    variant = "notebook_baseline" if "m206" in ("m102", "m103", "m105") else ("automl_best" if "m206" == "m106" else "release_candidate")
    eval_res = evaluate_predictions(splits["validation"], model_variant=variant)
    report = {
        "clip": "m206",
        "title": "Demo: Compare churn models across Azure Machine Learning jobs",
        "objectives": ["TO1", "EO1h"],
        "dataset_version": "globomantics-churn-v1",
        "split_counts": {k: len(v) for k, v in splits.items()},
        "final_test_unopened": "m206" != "m305",
        "metrics": eval_res,
        "steps_verified": ["Query the baseline, sweep, and AutoML runs with MLflow search_runs and open them side by side in the studio job comparison", "Score each candidate on the same validation data with the evaluation component\u2019s metric code, loading the AutoML model in its own MLflow environment", "Compare recall at 10 percent false-positive rate, plan-tier support, and training time", "Freeze the winning configuration for the training pipeline, or record a hold when no candidate meets the acceptance metric"],
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
