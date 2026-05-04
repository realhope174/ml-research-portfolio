import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
from feature_engineering import create_features
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load data
df = pd.read_csv('data/processed/cleaned_credit_data.csv')

# DIAGNOSTIC: Print columns to verify
print("Columns in dataset:")
print(df.columns.tolist())

# Continue with feature engineering
preprocessor, X, y = create_features(df)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Preprocess data
print("Preprocessing data...")
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Apply SMOTE to training data
print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_preprocessed, y_train)
print(f"Resampled training set size: {X_train_res.shape}")

# Train Logistic Regression
print("Training Logistic Regression...")
lr = LogisticRegression(class_weight='balanced', max_iter=2000)  # Increased max_iter
lr.fit(X_train_res, y_train_res)

# Train XGBoost
print("Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8
)
xgb_model.fit(X_train_res, y_train_res)

# Evaluation
print("Evaluating models...")
lr_probs = lr.predict_proba(X_test_preprocessed)[:, 1]
lr_auc = roc_auc_score(y_test, lr_probs)

xgb_probs = xgb_model.predict_proba(X_test_preprocessed)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_probs)

print(f"Logistic Regression AUC: {lr_auc:.4f}")
print(f"XGBoost AUC: {xgb_auc:.4f}")

# Save models and preprocessor
print("Saving models and preprocessor...")
joblib.dump(lr, 'models/logistic_regression.pkl')
joblib.dump(xgb_model, 'models/xgboost_model.pkl')
joblib.dump(preprocessor, 'models/preprocessor.pkl')  # Save preprocessor separately

print("Model training and saving completed successfully!")