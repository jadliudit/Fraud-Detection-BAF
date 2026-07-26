# Bank Account Fraud Detection — BAF Suite (NeurIPS 2022)

> Detecting fraudulent bank account applications under extreme class imbalance — with explainable predictions, cost-based decision thresholds, and a fairness audit across protected groups.

![Python](https://img.shields.io/badge/Python-3.12-1f2937?style=flat-square&logo=python&logoColor=4fd1c5)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-1f2937?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-1f2937?style=flat-square&logo=scikitlearn&logoColor=f7931e)
![SHAP](https://img.shields.io/badge/SHAP-explainability-1f2937?style=flat-square)
![Status](https://img.shields.io/badge/status-in%20progress-e9b949?style=flat-square)
![License](https://img.shields.io/badge/code-MIT-4fd1c5?style=flat-square)

---

## Overview

Financial institutions face a two-sided cost when screening new account applications. Approving a **fraudulent** application leads to direct losses and downstream misuse. Rejecting a **legitimate** applicant destroys a customer relationship and invites regulatory scrutiny over fairness.

With a fraud rate near **1%**, accuracy is a misleading metric — a model that approves every application is ~99% accurate and completely useless. This project frames fraud detection as a **business decision problem**: where should the approval threshold sit, given the relative cost of a missed fraud versus a wrongly rejected customer, and does that decision treat demographic groups fairly?

This is an **end-to-end modelling project** demonstrating honest handling of imbalanced data, explainable outputs, cost-aware evaluation, and fairness awareness.

## Business questions

1. Which application features most strongly signal fraud, and are they explainable to a risk or compliance stakeholder?
2. Where should the decision threshold sit to balance fraud loss against the cost of rejecting genuine customers?
3. Does the model's error rate differ across protected groups (age, employment, income), and can that gap be measured and reduced?

## Dataset

**Bank Account Fraud (BAF) Suite — NeurIPS 2022**, published by Feedzai. [Kaggle link](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)

- **Base variant:** ~1,000,000 applications, fraud prevalence ~1.1%
- **Six variants** (Base, Variant I–V) with controlled bias for fairness research
- **Protected attributes:** age group, employment status, % income
- **Privacy-preserving synthetic data** — generated from a real bank's account-opening fraud data using differential privacy, feature encoding, and a CTGAN generative model. This project uses the synthetic release, not raw records.
- **Licence:** CC BY-NC-ND 4.0 (academic / non-commercial). Data files are **not committed to this repository** — download from Kaggle into `data/`.

## Tech stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.12 |
| Data | pandas, NumPy |
| Visualisation | Matplotlib, seaborn |
| Modelling | scikit-learn (Logistic Regression baseline), XGBoost |
| Imbalance | imbalanced-learn (SMOTE), class weighting |
| Explainability | SHAP |
| Serving | FastAPI + Uvicorn, Streamlit |
| Tooling | Jupyter, Git/GitHub |

**Design choice:** gradient-boosted trees over deep learning. On tabular data with mixed categorical and numeric features, boosting is the appropriate and better-performing tool — a deliberate decision, not a default.

## Methodology

1. **EDA** — quantify imbalance, profile features, locate `-1`-encoded missing values
2. **Preprocessing** — train/test split *before* any resampling to prevent leakage
3. **Imbalance handling** — compare SMOTE against class weighting
4. **Modelling** — Logistic Regression baseline, then XGBoost
5. **Evaluation** — precision/recall, PR-AUC, ROC-AUC, confusion matrix
6. **Cost-based threshold tuning** — optimise the decision point on business cost
7. **Explainability** — SHAP global and local, mapped to fraud typologies
8. **Fairness audit** — error-rate parity across protected groups
9. **Serving** — FastAPI scoring endpoint and a Streamlit demo

## Results

> _In progress — metrics to be added once modelling is complete._

| Model | PR-AUC | ROC-AUC | Recall @ fixed FPR | Notes |
|-------|--------|---------|--------------------|-------|
| Logistic Regression (baseline) | _tbd_ | _tbd_ | _tbd_ | |
| XGBoost | _tbd_ | _tbd_ | _tbd_ | |

## Scope — what this is, and what it is not

This is a **modelling** project on a static, publicly released dataset. It does **not** claim production deployment. A production fraud system at a bank would additionally require real-time low-latency scoring, streaming ingestion, model-drift monitoring, scheduled retraining, a feature store, model-governance sign-off, and handling of adversarial adaptation as fraudsters change behaviour. Stating this explicitly is intentional: the work here is a genuine but partial slice of the full fraud-detection lifecycle.

## Repository structure