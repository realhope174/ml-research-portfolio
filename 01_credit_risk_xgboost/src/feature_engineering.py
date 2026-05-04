import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def create_features(df):
    # Diagnostic: Print columns to verify
    print("Columns in feature_engineering:")
    print(df.columns.tolist())
    
    # Create new features using ACTUAL columns in your dataset
    df['debt_to_income'] = df['loan_amnt'] / (df['person_income'] + 1)
    df['income_to_emp_length'] = df['person_income'] / (df['person_emp_length'] + 1)
    
    # Create credit history length feature (using actual column)
    df['credit_history_length'] = df['cb_person_cred_hist_length']
    
    # Binning numerical features
    df['income_group'] = pd.cut(df['person_income'],
                               bins=[0, 30000, 60000, 100000, float('inf')],
                               labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Prepare for modeling
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    
    # Preprocessing pipeline
    categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'income_group']
    numerical_features = ['person_age', 'person_emp_length', 'loan_amnt', 
                         'loan_int_rate', 'debt_to_income', 
                         'income_to_emp_length', 'credit_history_length']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    return preprocessor, X, y