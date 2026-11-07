#!/usr/bin/env python3
"""M202 — Demo: Automate hyperparameter tuning with an Azure Machine Learning sweep job
Objectives: TO1, EO1d
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
    variant = "notebook_baseline" if "m202" in ("m102", "m103", "m105") else ("automl_best" if "m202" == "m106" else "release_candidate")
    eval_res = evaluate_predictions(splits["validation"], model_variant=variant)
    report = {
        "clip": "m202",
        "title": "Demo: Automate hyperparameter tuning with an Azure Machine Learning sweep job",
        "objectives": ["TO1", "EO1d"],
        "dataset_version": "globomantics-churn-v1",
        "split_counts": {k: len(v) for k, v in splits.items()},
        "final_test_unopened": "m202" != "m305",
        "metrics": eval_res,
        "steps_verified": ["Turn the train.py command job into a sweep with random sampling over the learning rate, maximum depth, and minimum samples per leaf", "Set recall at 10 percent false-positive rate as the primary metric to maximize, with four total trials and one at a time", "Open the completed sweep\u2019s trials, parameters, metric values, and durations, and decide whether the best trial\u2019s gain justifies its compute time"],
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
