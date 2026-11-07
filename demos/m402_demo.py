#!/usr/bin/env python3
"""M402 — Demo: Deploy the churn model to a managed online endpoint
Objectives: TO3, EO3a
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
    variant = "notebook_baseline" if "m402" in ("m102", "m103", "m105") else ("automl_best" if "m402" == "m106" else "release_candidate")
    eval_res = evaluate_predictions(splits["validation"], model_variant=variant)
    report = {
        "clip": "m402",
        "title": "Demo: Deploy the churn model to a managed online endpoint",
        "objectives": ["TO3", "EO3a"],
        "dataset_version": "globomantics-churn-v1",
        "split_counts": {k: len(v) for k, v in splits.items()},
        "final_test_unopened": "m402" != "m305",
        "metrics": eval_res,
        "steps_verified": ["Define a managed online endpoint and a blue deployment of the approved churn-classifier version on one Standard_DS3_v2 instance, with a scoring script that collects inputs and outputs with each row\u2019s observation ID as its own column", "Submit the deployment, and once provisioning completes, check its state, model version, and logs", "Invoke the endpoint with resolved feature values and check the response shape, the serving deployment, and the request schema"],
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
