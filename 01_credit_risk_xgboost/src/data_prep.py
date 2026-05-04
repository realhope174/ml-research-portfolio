# src/data_prep.py
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data():
    # Load raw data
    df = pd.read_csv('data/raw/credit_risk_dataset.csv')
    
    # Basic cleaning
    df = df.dropna(subset=['person_emp_length'])
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())
    
    # Save cleaned data
    df.to_csv('data/processed/cleaned_credit_data.csv', index=False)
    
    # Create SQL database
    engine = create_engine('sqlite:///data/credit_risk.db')
    df.to_sql('loan_applications', engine, if_exists='replace', index=False)
    
    return df

if __name__ == "__main__":
    df = load_data()
    print(f"Data loaded! Shape: {df.shape}")