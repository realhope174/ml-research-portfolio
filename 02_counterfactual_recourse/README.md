# Counterfactual Recourse for Credit Decisions

**Author:** Hope Kwadzo Dzamesi  
**Dataset:** German Credit Dataset (UCI Machine Learning Repository)  
**Tools:** Python, XGBoost, pandas, NumPy, Matplotlib

## Overview

This project demonstrates counterfactual recourse in a credit decision system.
Rather than simply telling an applicant they were rejected, the system identifies
the minimal, actionable changes to their financial profile that would flip the
model decision from rejected to approved.

## Key Results

- XGBoost classifier trained on 1,000 applicants — ROC-AUC: **0.7998**
- Counterfactual found for a rejected applicant with **1 feature change**
- Approval probability shifted from **0.37 → 0.58**, crossing the decision threshold

## Connection to Research Literature

This work connects directly to two active research areas:

**Model Multiplicity / Rashomon Set:** Different models with near-identical
predictive accuracy may generate very different counterfactual recommendations
for the same applicant. This instability raises serious fairness concerns in
high-stakes financial decisions.

**Decision-Focused Learning:** Rather than optimising for predictive accuracy
alone and handling recourse as an afterthought, decision-focused learning
incorporates recourse quality directly into the training objective — producing
models that are both accurate and actionable.

## Files

- `counterfactual_recourse.ipynb` — Main analysis notebook
- `counterfactual_recourse.py` — Python script version
- `counterfactual_recourse_results.png` — Visualisation output

## How to Run

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
jupyter notebook counterfactual_recourse.ipynb
```
