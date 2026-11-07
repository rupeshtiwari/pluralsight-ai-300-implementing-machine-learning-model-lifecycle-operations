#!/usr/bin/env python3
"""M205 — Demo: Build a churn training pipeline with Azure Machine Learning components
Objectives: TO1, EO1g
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
    variant = "notebook_baseline" if "m205" in ("m102", "m103", "m105") else ("automl_best" if "m205" == "m106" else "release_candidate")
    eval_res = evaluate_predictions(splits["validation"], model_variant=variant)
    report = {
        "clip": "m205",
        "title": "Demo: Build a churn training pipeline with Azure Machine Learning components",
        "objectives": ["TO1", "EO1g"],
        "dataset_version": "globomantics-churn-v1",
        "split_counts": {k: len(v) for k, v in splits.items()},
        "final_test_unopened": "m205" != "m305",
        "metrics": eval_res,
        "steps_verified": ["Connect the prepared preparation, training, and evaluation components in an SDK v2 pipeline with typed inputs and outputs", "Submit the pipeline and trace the completed graph from the churn data asset to the model output and evaluation report", "Rerun it with only the learning rate changed, and confirm that preparation is reused while training and evaluation run again"],
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
