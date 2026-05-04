import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import roc_curve, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split  # ADD THIS IMPORT
import seaborn as sns
import joblib
from pathlib import Path

# Set up paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data/processed'
IMAGES_DIR = BASE_DIR / 'images'

# Load best model and preprocessor
print("Loading models...")
xgb_model = joblib.load(MODELS_DIR / 'xgboost_model.pkl')
preprocessor = joblib.load(MODELS_DIR / 'preprocessor.pkl')

# Load data
print("Loading data...")
df = pd.read_csv(DATA_DIR / 'cleaned_credit_data.csv')

# Create features and split
from feature_engineering import create_features
_, X, y = create_features(df)

# Split data with the same random_state used in modeling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocess test data
print("Preprocessing test data...")
X_test_preprocessed = preprocessor.transform(X_test)

# ROC Curve
print("Generating ROC curve...")
fpr, tpr, _ = roc_curve(y_test, xgb_model.predict_proba(X_test_preprocessed)[:,1])
plt.figure(figsize=(10,6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, xgb_model.predict_proba(X_test_preprocessed)[:,1]):.2f}')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig(IMAGES_DIR / 'roc_curve.png')
plt.close()

# SHAP Analysis (only for XGBoost)
print("Generating SHAP summary...")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_preprocessed)

plt.figure()
shap.summary_plot(shap_values, X_test_preprocessed, feature_names=preprocessor.get_feature_names_out(), show=False)
plt.tight_layout()
plt.savefig(IMAGES_DIR / 'shap_summary.png')
plt.close()

# Confusion Matrix
print("Generating confusion matrix...")
y_pred = xgb_model.predict(X_test_preprocessed)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(IMAGES_DIR / 'confusion_matrix.png')
plt.close()

# Precision-Recall Curve
print("Generating precision-recall curve...")
precision, recall, _ = precision_recall_curve(y_test, xgb_model.predict_proba(X_test_preprocessed)[:,1])
plt.figure(figsize=(10,6))
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig(IMAGES_DIR / 'precision_recall_curve.png')
plt.close()

print("Evaluation completed! Check the images directory.")