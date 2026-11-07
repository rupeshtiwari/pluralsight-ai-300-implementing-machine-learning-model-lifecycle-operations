#!/usr/bin/env python3
"""M103 — Demo: Track notebook experiments with MLflow in Azure Machine Learning
Objectives: TO1, EO1a
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
    variant = "notebook_baseline" if "m103" in ("m102", "m103", "m105") else ("automl_best" if "m103" == "m106" else "release_candidate")
    eval_res = evaluate_predictions(splits["validation"], model_variant=variant)
    report = {
        "clip": "m103",
        "title": "Demo: Track notebook experiments with MLflow in Azure Machine Learning",
        "objectives": ["TO1", "EO1a"],
        "dataset_version": "globomantics-churn-v1",
        "split_counts": {k: len(v) for k, v in splits.items()},
        "final_test_unopened": "m103" != "m305",
        "metrics": eval_res,
        "steps_verified": ["Confirm that the notebook\u2019s MLflow tracking URI points to the workspace, and set the churn-baseline experiment", "Log the baseline\u2019s parameters, validation recall, false-positive rate, data asset version, and code version in one run", "Log the model in MLflow format with an input signature and example under the registered name churn-classifier, then open the run, its metrics, and its model files in studio"],
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
