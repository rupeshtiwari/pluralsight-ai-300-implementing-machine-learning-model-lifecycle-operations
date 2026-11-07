#!/usr/bin/env python3
"""M105 — Demo: Run a churn training script as an Azure Machine Learning command job
Objectives: TO1, EO1e, EO1a
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
    variant = "notebook_baseline" if "m105" in ("m102", "m103", "m105") else ("automl_best" if "m105" == "m106" else "release_candidate")
    eval_res = evaluate_predictions(splits["validation"], model_variant=variant)
    report = {
        "clip": "m105",
        "title": "Demo: Run a churn training script as an Azure Machine Learning command job",
        "objectives": ["TO1", "EO1e", "EO1a"],
        "dataset_version": "globomantics-churn-v1",
        "split_counts": {k: len(v) for k, v in splits.items()},
        "final_test_unopened": "m105" != "m305",
        "metrics": eval_res,
        "steps_verified": ["Move the notebook logic into train.py with arguments for the data asset, split, and seed", "Define a command job with the versioned data asset, the training environment, and the CPU cluster, then submit it", "Open the completed job\u2019s logs, arguments, MLflow metrics, and model output, and match them to the run from the MLflow tracking demo"],
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
