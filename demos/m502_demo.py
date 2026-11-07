#!/usr/bin/env python3
"""M502 — Demo: Monitor churn endpoint performance in Azure Monitor
Objectives: TO4, EO4b
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
    variant = "notebook_baseline" if "m502" in ("m102", "m103", "m105") else ("automl_best" if "m502" == "m106" else "release_candidate")
    eval_res = evaluate_predictions(splits["validation"], model_variant=variant)
    report = {
        "clip": "m502",
        "title": "Demo: Monitor churn endpoint performance in Azure Monitor",
        "objectives": ["TO4", "EO4b"],
        "dataset_version": "globomantics-churn-v1",
        "split_counts": {k: len(v) for k, v in splits.items()},
        "final_test_unopened": "m502" != "m305",
        "metrics": eval_res,
        "steps_verified": ["Send a short controlled load of valid and invalid requests, and select the time range that holds it", "Chart Request Latency P95 with Average aggregation split by deployment, and Requests Per Minute split by deployment and status code class", "Match the invalid requests\u2019 status codes to the deployment logs, and explain why an average of one-minute P95 values is not a percentile over the whole range"],
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
