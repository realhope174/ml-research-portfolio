import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os

# Set up paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / 'models'

# Load models and preprocessor
try:
    preprocessor = joblib.load(MODELS_DIR / 'preprocessor.pkl')
    model = joblib.load(MODELS_DIR / 'xgboost_model.pkl')
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

st.title('Credit Risk Assessment Dashboard')
st.subheader('Predict Loan Default Probability')

# Input form with all required features
with st.form("applicant_info"):
    col1, col2 = st.columns(2)
    
    with col1:
        person_age = st.slider('Age', 18, 100, 30)
        person_income = st.number_input('Annual Income ($)', 0, 1000000, 50000)
        person_emp_length = st.slider('Employment Length (years)', 0, 50, 5)
        person_home_ownership = st.selectbox('Home Ownership', 
                                            ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
        loan_intent = st.selectbox('Loan Purpose', 
                                  ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT'])
        
    with col2:
        loan_amnt = st.number_input('Loan Amount ($)', 0, 100000, 10000)
        loan_int_rate = st.slider('Interest Rate (%)', 0.0, 30.0, 10.0)
        cb_person_cred_hist_length = st.slider('Credit History Length (years)', 0, 50, 5)
        cb_person_default_on_file = st.selectbox('Previous Default?', ['N', 'Y'])
        loan_grade = st.select_slider('Loan Grade', options=['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    
    submitted = st.form_submit_button("Assess Risk")

# Create feature dictionary with ALL required features
feature_dict = {
    'person_age': person_age,
    'person_income': person_income,
    'person_home_ownership': person_home_ownership,
    'person_emp_length': person_emp_length,
    'loan_intent': loan_intent,
    'loan_grade': loan_grade,
    'loan_amnt': loan_amnt,
    'loan_int_rate': loan_int_rate,
    'cb_person_default_on_file': cb_person_default_on_file,
    'cb_person_cred_hist_length': cb_person_cred_hist_length,
    # These will be calculated in create_features
    'loan_status': 0,  # Dummy value, not used
    'loan_percent_income': loan_amnt / (person_income + 1),  # Actual column in dataset
}

# Calculate derived features used in our model
feature_dict['debt_to_income'] = loan_amnt / (person_income + 1)
feature_dict['income_to_emp_length'] = person_income / (person_emp_length + 1)
feature_dict['credit_history_length'] = cb_person_cred_hist_length

# Create income group
income_group = 'Low' if person_income < 30000 else \
               'Medium' if person_income < 60000 else \
               'High' if person_income < 100000 else 'Very High'
feature_dict['income_group'] = income_group

if submitted:
    try:
        # Create dataframe
        input_df = pd.DataFrame([feature_dict])
        
        # Preprocess input
        input_processed = preprocessor.transform(input_df)
        
        # Predict
        prob_default = model.predict_proba(input_processed)[0][1]
        risk_score = int((1 - prob_default) * 850)
        
        # Display results
        st.subheader("Risk Assessment Result")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Default Probability", f"{prob_default:.1%}")
        
        with col2:
            st.metric("Credit Score", f"{risk_score}")
        
        with col3:
            if prob_default < 0.1:
                st.success("Low Risk")
            elif prob_default < 0.3:
                st.warning("Medium Risk")
            else:
                st.error("High Risk")
        
        # Risk tier explanation
        st.progress(int(prob_default * 100))
        st.caption("Risk Probability Indicator")
        
        # Recommendation
        if prob_default < 0.2:
            st.success("Recommendation: **Approve Loan**")
        elif prob_default < 0.4:
            st.warning("Recommendation: **Approve with Higher Interest**")
            # Calculate risk-based interest rate
            base_rate = 5.0
            risk_adjustment = prob_default * 20
            st.info(f"Suggested interest rate: {base_rate + risk_adjustment:.1f}%")
        else:
            st.error("Recommendation: **Reject Application**")
            
        # Show key risk factors
        st.subheader("Key Risk Factors")
        factors = {
            "High debt-to-income ratio": feature_dict['debt_to_income'] > 0.4,
            "Short employment history": person_emp_length < 2,
            "Poor credit history": cb_person_default_on_file == 'Y',
            "High interest rate": loan_int_rate > 15,
        }
        
        for factor, condition in factors.items():
            if condition:
                st.error(f"⚠️ {factor}")
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.error("Please check your input values")

# Add footer
st.markdown("---")
st.caption("Credit Risk Scoring Model v1.0 | Developed for Portfolio Project")