import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import os
import sys

# Get current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct absolute path to database
db_path = os.path.normpath(os.path.join(script_dir, '..', 'data', 'credit_risk.db'))
print(f"Database path: {db_path}")

# Verify database exists
if not os.path.exists(db_path):
    print(f"Error: Database file not found at {db_path}")
    print("Please run data_prep.py first")
    sys.exit(1)

# Connect to SQL database
engine = create_engine(f'sqlite:///{db_path}')

try:
    df = pd.read_sql_table('loan_applications', engine)
    print("Database connection successful!")
    print(f"Loaded {len(df)} records")
    
    # ... rest of your EDA code ...
    
except Exception as e:
    print(f"Database error: {str(e)}")
    sys.exit(1)