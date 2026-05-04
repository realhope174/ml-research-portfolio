# Counterfactual Recourse for Credit Decisions
# Author: Hope Kwadzo Dzamesi
# Description: This project demonstrates counterfactual recourse in credit
# decision systems. Given a rejected loan applicant, we identify the minimal
# changes to their features that would flip the model decision to approved.
# This connects directly to the model multiplicity and causal recourse
# literature in explainable AI for financial services.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
# German Credit Dataset (UCI Machine Learning Repository)
# 1000 applicants, 20 features, binary outcome: 1=Good credit, 2=Bad credit
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

columns = [
    "checking_account", "duration", "credit_history", "purpose", "credit_amount",
    "savings_account", "employment", "installment_rate", "personal_status",
    "other_debtors", "residence_since", "property", "age", "other_installments",
    "housing", "existing_credits", "job", "dependents", "telephone", "foreign_worker",
    "target"
]

df = pd.read_csv(url, sep=" ", header=None, names=columns)

# Recode target: 1=Good (approved), 0=Bad (rejected)
df["target"] = df["target"].map({1: 1, 2: 0})

print("Dataset shape:", df.shape)
print("\nTarget distribution:")
print(df["target"].value_counts())

# ── 2. PREPROCESS ─────────────────────────────────────────────────────────────
# Encode categorical features
df_encoded = df.copy()
le = LabelEncoder()
cat_cols = df.select_dtypes(include="object").columns

for col in cat_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])

X = df_encoded.drop("target", axis=1)
y = df_encoded["target"]

feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples")

# ── 3. TRAIN XGBOOST MODEL ────────────────────────────────────────────────────
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nModel Performance:")
print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# ── 4. SELECT A REJECTED APPLICANT ───────────────────────────────────────────
# Find applicants in the test set that the model rejected
rejected_mask = y_pred == 0
rejected_indices = X_test[rejected_mask].index

# Pick the first rejected applicant
applicant_idx = rejected_indices[0]
applicant = X_test.loc[[applicant_idx]].copy()

print(f"\nApplicant {applicant_idx} — REJECTED")
print(applicant.T.rename(columns={applicant_idx: "Value"}))

# ── 5. COUNTERFACTUAL SEARCH ──────────────────────────────────────────────────
# Strategy: Greedy search over mutable features
# We perturb features one at a time and find the combination that flips
# the decision with the smallest total change (L1 distance)
#
# Mutable features = features an applicant can realistically change
# Immutable features = age, foreign_worker, personal_status (cannot be changed)

mutable_features = [
    "duration", "credit_amount", "installment_rate",
    "savings_account", "employment", "existing_credits"
]

def predict_proba(x):
    return model.predict_proba(x)[0][1]  # probability of approval

original_prob = predict_proba(applicant)
print(f"\nOriginal approval probability: {original_prob:.4f}")

# Search: perturb each mutable feature across a range of values
# and find the set of changes that flips the decision above 0.5
best_counterfactual = None
best_distance = np.inf
approval_threshold = 0.5

# Generate candidate counterfactuals by perturbing features
np.random.seed(42)
n_samples = 5000
counterfactuals = []

for _ in range(n_samples):
    candidate = applicant.copy()
    changes = {}

    # Randomly perturb a random subset of mutable features
    n_features_to_change = np.random.randint(1, len(mutable_features) + 1)
    features_to_change = np.random.choice(mutable_features, n_features_to_change, replace=False)

    for feat in features_to_change:
        # Sample from the training distribution of that feature
        new_val = X_train[feat].sample(1).values[0]
        candidate[feat] = new_val
        changes[feat] = (applicant[feat].values[0], new_val)

    prob = predict_proba(candidate)

    if prob >= approval_threshold:
        # L1 distance: count how many features changed
        distance = len(features_to_change)
        counterfactuals.append({
            "candidate": candidate,
            "changes": changes,
            "prob": prob,
            "distance": distance
        })

# Sort by fewest changes first
counterfactuals.sort(key=lambda x: (x["distance"], -x["prob"]))

if counterfactuals:
    best = counterfactuals[0]
    best_counterfactual = best["candidate"]
    best_changes = best["changes"]
    best_prob = best["prob"]
    print(f"\nCounterfactual found with {best['distance']} feature change(s)")
    print(f"New approval probability: {best_prob:.4f}")
    print("\nRequired changes:")
    for feat, (old_val, new_val) in best_changes.items():
        print(f"  {feat}: {old_val:.2f} → {new_val:.2f}")
else:
    print("\nNo counterfactual found within search budget.")

# ── 6. VISUALISE RESULTS ──────────────────────────────────────────────────────
if counterfactuals:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Feature comparison — original vs counterfactual
    changed_feats = list(best_changes.keys())
    original_vals = [best_changes[f][0] for f in changed_feats]
    new_vals = [best_changes[f][1] for f in changed_feats]

    x = np.arange(len(changed_feats))
    width = 0.35

    axes[0].bar(x - width/2, original_vals, width, label="Original (Rejected)",
                color="#d9534f", alpha=0.85)
    axes[0].bar(x + width/2, new_vals, width, label="Counterfactual (Approved)",
                color="#5cb85c", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(changed_feats, rotation=30, ha="right")
    axes[0].set_title("Feature Changes: Original vs Counterfactual", fontsize=13)
    axes[0].set_ylabel("Feature Value")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Plot 2: Approval probability shift
    probs = [original_prob, best_prob]
    labels = ["Original\n(Rejected)", "Counterfactual\n(Approved)"]
    colors = ["#d9534f", "#5cb85c"]

    bars = axes[1].bar(labels, probs, color=colors, alpha=0.85, width=0.4)
    axes[1].axhline(y=0.5, color="black", linestyle="--", linewidth=1.2,
                    label="Decision threshold (0.5)")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Approval Probability")
    axes[1].set_title("Approval Probability: Before and After Recourse", fontsize=13)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    for bar, prob in zip(bars, probs):
        axes[1].text(bar.get_x() + bar.get_width()/2, prob + 0.02,
                     f"{prob:.3f}", ha="center", fontsize=12, fontweight="bold")

    plt.suptitle(
        "Counterfactual Recourse in Credit Decision Systems\n"
        "Minimal feature changes to flip a rejection to approval",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig("counterfactual_recourse_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nPlot saved as counterfactual_recourse_results.png")

# ── 7. INTERPRETATION ─────────────────────────────────────────────────────────
print("\n" + "="*65)
print("INTERPRETATION")
print("="*65)
print("""
This analysis demonstrates counterfactual recourse in a credit
decision system. Rather than simply telling an applicant they
were rejected, the model identifies the minimal, actionable
changes to their financial profile that would result in approval.

This approach addresses a core challenge in responsible AI:
decisions should not only be accurate but also explainable and
actionable for the individuals affected by them.

Connection to model multiplicity:
  Different models in the Rashomon set — models with near-identical
  predictive accuracy — may generate very different counterfactual
  recommendations for the same applicant. A model that recommends
  'reduce loan duration' and another that recommends 'increase savings'
  may both be equally accurate, yet give conflicting guidance.
  This instability is a key motivation for decision-focused learning,
  which incorporates recourse quality directly into the training objective.
""")
