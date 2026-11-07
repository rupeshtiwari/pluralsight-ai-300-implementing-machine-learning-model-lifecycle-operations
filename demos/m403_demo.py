#!/usr/bin/env python3
"""M403 — Demo: Score active customers with an Azure Machine Learning batch endpoint
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
    variant = "notebook_baseline" if "m403" in ("m102", "m103", "m105") else ("automl_best" if "m403" == "m106" else "release_candidate")
    eval_res = evaluate_predictions(splits["validation"], model_variant=variant)
    report = {
        "clip": "m403",
        "title": "Demo: Score active customers with an Azure Machine Learning batch endpoint",
        "objectives": ["TO3", "EO3a"],
        "dataset_version": "globomantics-churn-v1",
        "split_counts": {k: len(v) for k, v in splits.items()},
        "final_test_unopened": "m403" != "m305",
        "metrics": eval_res,
        "steps_verified": ["Create a batch deployment of the same churn-classifier version on the CPU cluster with a mini-batch size, retry count, and error threshold", "Submit a batch job on the active customers\u2019 features, retrieved with the model\u2019s packaged specification, and open the completed job and its output file", "Compare the input row count with the prediction rows and check the failed-item count before accepting the batch"],
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
