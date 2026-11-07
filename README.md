# AI-300: Implementing Machine Learning Model Lifecycle and Operations — Learner Lab Repository

Hands-on lab exercises, synthetic Globomantics subscription-churn dataset generator, Azure Machine Learning SDK v2 scripts, MLflow tracking & registry workflows, Responsible AI evaluation, managed online/batch endpoints, and production model monitoring for the Pluralsight course **AI-300: Implementing Machine Learning Model Lifecycle and Operations**.

## Quick Start

```bash
python3 -m pip install -e .
./run-all-demo-preflights.sh
pytest -q
```

## Module & Demo Directory (All 20 Hands-On Demos)

| Clip ID | Demo Title | Learning Objectives | Lab Command |
| :--- | :--- | :--- | :--- |
| `M1-C2` (`m102`) | Demo: Explore the churn data in an Azure Machine Learning notebook | `TO1, EO1c` | `python3 demos/m102_demo.py` |
| `M1-C3` (`m103`) | Demo: Track notebook experiments with MLflow in Azure Machine Learning | `TO1, EO1a` | `python3 demos/m103_demo.py` |
| `M1-C5` (`m105`) | Demo: Run a churn training script as an Azure Machine Learning command job | `TO1, EO1e, EO1a` | `python3 demos/m105_demo.py` |
| `M1-C6` (`m106`) | Demo: Explore churn models with Azure automated machine learning | `TO1, EO1b` | `python3 demos/m106_demo.py` |
| `M2-C2` (`m202`) | Demo: Automate hyperparameter tuning with an Azure Machine Learning sweep job | `TO1, EO1d` | `python3 demos/m202_demo.py` |
| `M2-C3` (`m203`) | Demo: Run distributed PyTorch training across two Azure Machine Learning nodes | `TO1, EO1f` | `python3 demos/m203_demo.py` |
| `M2-C5` (`m205`) | Demo: Build a churn training pipeline with Azure Machine Learning components | `TO1, EO1g` | `python3 demos/m205_demo.py` |
| `M2-C6` (`m206`) | Demo: Compare churn models across Azure Machine Learning jobs | `TO1, EO1h` | `python3 demos/m206_demo.py` |
| `M3-C2` (`m302`) | Demo: Package a feature retrieval specification with the churn model | `TO2, EO2a` | `python3 demos/m302_demo.py` |
| `M3-C3` (`m303`) | Demo: Register the churn model as an MLflow model in Azure Machine Learning | `TO2, EO2b, EO2a` | `python3 demos/m303_demo.py` |
| `M3-C5` (`m305`) | Demo: Evaluate the churn model with the Azure Machine Learning Responsible AI dashboard | `TO2, EO2c` | `python3 demos/m305_demo.py` |
| `M3-C6` (`m306`) | Demo: Archive and restore churn model versions in Azure Machine Learning | `TO2, EO2d` | `python3 demos/m306_demo.py` |
| `M4-C2` (`m402`) | Demo: Deploy the churn model to a managed online endpoint | `TO3, EO3a` | `python3 demos/m402_demo.py` |
| `M4-C3` (`m403`) | Demo: Score active customers with an Azure Machine Learning batch endpoint | `TO3, EO3a` | `python3 demos/m403_demo.py` |
| `M4-C5` (`m405`) | Demo: Test and troubleshoot the churn online endpoint | `TO3, EO3b` | `python3 demos/m405_demo.py` |
| `M4-C6` (`m406`) | Demo: Roll out a churn candidate with mirrored and weighted traffic | `TO3, EO3c` | `python3 demos/m406_demo.py` |
| `M5-C2` (`m502`) | Demo: Monitor churn endpoint performance in Azure Monitor | `TO4, EO4b` | `python3 demos/m502_demo.py` |
| `M5-C3` (`m503`) | Demo: Detect data drift with Azure Machine Learning model monitoring | `TO4, EO4a` | `python3 demos/m503_demo.py` |
| `M5-C4` (`m504`) | Demo: Monitor labeled churn model performance | `TO4, EO4b` | `python3 demos/m504_demo.py` |
| `M5-C5` (`m505`) | Demo: Trigger an alert and a gated retraining pipeline from model monitoring | `TO4, EO4c` | `python3 demos/m505_demo.py` |

## Environment & Stack

- **Python**: `3.12`
- **Azure ML & MLflow Stack**: `azure-ai-ml==1.35.2`, `azure-identity==1.27.0`, `mlflow-skinny==3.15.0`, `azureml-mlflow==1.62.0.post6`, `scikit-learn==1.9.1`, `pandas==3.1.0`, `numpy==2.5.4`, `torch==2.14.1`, `azureml-featurestore==1.2.2`, `azureml-inference-server-http==1.5.1`, `azureml-ai-monitoring==1.0.0`, `azure-functions==1.25.0`.
