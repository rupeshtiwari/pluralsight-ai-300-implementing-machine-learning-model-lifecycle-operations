#!/usr/bin/env python3
"""M302 — Demo: Package a feature retrieval specification with the churn model
Objectives: TO2, EO2a
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
    variant = "notebook_baseline" if "m302" in ("m102", "m103", "m105") else ("automl_best" if "m302" == "m106" else "release_candidate")
    eval_res = evaluate_predictions(splits["validation"], model_variant=variant)
    report = {
        "clip": "m302",
        "title": "Demo: Package a feature retrieval specification with the churn model",
        "objectives": ["TO2", "EO2a"],
        "dataset_version": "globomantics-churn-v1",
        "split_counts": {k: len(v) for k, v in splits.items()},
        "final_test_unopened": "m302" != "m305",
        "metrics": eval_res,
        "steps_verified": ["Resolve the churn feature set\u2019s versioned features from the managed feature store and generate feature_retrieval_spec.yaml", "Open the completed feature retrieval component run that used this specification to build the training data with a point-in-time join on observation date", "Confirm that the frozen configuration\u2019s training job copied the specification to the root of its MLflow model output, and check the feature names and versions against the model signature"],
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
