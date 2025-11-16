# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page config
st.set_page_config(page_title="EMIPredict AI", layout="wide", initial_sidebar_state="expanded")

# --- Load Models ---
@st.cache_resource
def load_models():
    """Loads the trained models from disk."""
    try:
        model_class = joblib.load('classification_model.pkl')
        model_reg = joblib.load('regression_model.pkl')
        return model_class, model_reg
    except FileNotFoundError:
        st.error("Model files not found! Please run the `train.py` script first to generate them.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

model_class, model_reg = load_models()

# --- Feature Engineering Function ---
# This MUST be identical to the one in train.py
def feature_engineer(data):
    expense_cols = ['monthly_rent', 'school_fees', 'college_fees', 'travel_expenses', 'groceries_utilities', 'other_monthly_expenses']
    data['total_expenses'] = data[expense_cols].sum(axis=1)
    data['total_debt'] = data['total_expenses'] + data['current_emi_amount']
    data['dti_ratio'] = data['total_debt'] / (data['monthly_salary'] + 1e-6)
    data['total_savings'] = data['bank_balance'] + data['emergency_fund']
    data['savings_to_income_ratio'] = data['total_savings'] / (data['monthly_salary'] + 1e-6)
    data['disposable_income'] = data['monthly_salary'] - data['total_debt']
    data['loan_affordability_ratio'] = data['requested_amount'] / (data['disposable_income'] + 1e-6)
    data['financial_risk_score'] = (data['credit_score'] / 850) + (data['savings_to_income_ratio'] * 0.5) - (data['dti_ratio'] * 0.5)
    data['age_x_salary'] = data['age'] * data['monthly_salary']
    data['credit_score_x_dti'] = data['credit_score'] * data['dti_ratio']
    data.replace([np.inf, -np.inf], 0, inplace=True)
    data.fillna(0, inplace=True)
    return data

# --- Multi-Page Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Live Risk Assessment"])

# ======================================================================================
# PAGE 1: LIVE RISK ASSESSMENT (Step 6)
# ======================================================================================
if page == "Live Risk Assessment":
    st.title("EMIPredict AI - Intelligent Financial Risk Assessment")
    st.markdown("This platform uses Machine Learning to predict EMI eligibility (Classification) and the maximum safe EMI amount (Regression).")
    
    st.sidebar.header("Enter Financial Details")
    
    with st.sidebar.form(key='input_form'):
        # Personal Demographics
        with st.expander("Personal Demographics", expanded=True):
            age = st.slider("Age", 25, 60, 35)
            gender = st.selectbox("Gender", ("Male", "Female"))
            marital_status = st.selectbox("Marital Status", ("Single", "Married"))
            education = st.selectbox("Education", ("High School", "Graduate", "Post Graduate", "Professional"))

        # Employment and Income
        with st.expander("Employment and Income", expanded=True):
            monthly_salary = st.number_input("Monthly Salary (INR)", 15000, 200000, 50000, step=1000)
            employment_type = st.selectbox("Employment Type", ("Private", "Government", "Self-employed"))
            years_of_employment = st.slider("Years of Employment", 0, 40, 5)
            company_type = st.selectbox("Company Type", ("Startup", "SME", "MNC", "Public Sector"))

        # Housing and Family
        with st.expander("Housing and Family"):
            house_type = st.selectbox("House Type", ("Rented", "Own", "Family"))
            monthly_rent = st.number_input("Monthly Rent", 0, 50000, 10000)
            family_size = st.slider("Family Size", 1, 10, 3)
            dependents = st.slider("Dependents", 0, 5, 1)

        # Monthly Financial Obligations
        with st.expander("Monthly Financial Obligations"):
            school_fees = st.number_input("School Fees", 0, 30000, 0)
            college_fees = st.number_input("College Fees", 0, 50000, 0)
            travel_expenses = st.number_input("Travel Expenses", 0, 20000, 3000)
            groceries_utilities = st.number_input("Groceries & Utilities", 0, 30000, 5000)
            other_monthly_expenses = st.number_input("Other Expenses", 0, 30000, 2000)

        # Financial Status and Credit History
        with st.expander("Financial Status and Credit History"):
            existing_loans = st.selectbox("Existing Loans?", ("Yes", "No"))
            current_emi_amount = st.number_input("Current EMI Amount", 0, 100000, 0)
            credit_score = st.slider("Credit Score", 300, 850, 750)
            bank_balance = st.number_input("Bank Balance", 0, 5000000, 50000)
            emergency_fund = st.number_input("Emergency Fund", 0, 1000000, 25000)
        
        # Loan Application Details
        with st.expander("Loan Application Details", expanded=True):
            emi_scenario = st.selectbox("EMI Scenario", ("E-commerce Shopping EMI", "Home Appliances EMI", "Vehicle EMI", "Personal Loan EMI", "Education EMI"))
            requested_amount = st.number_input("Requested Loan Amount", 10000, 1500000, 100000)
            requested_tenure = st.slider("Requested Tenure (months)", 3, 84, 12)
        
        submit_button = st.form_submit_button(label='Assess Financial Risk', use_container_width=True)

    # --- Main Page for Predictions ---
    if submit_button:
        # Create a DataFrame from the inputs
        data = {
            'age': age, 'gender': gender, 'marital_status': marital_status, 'education': education,
            'monthly_salary': monthly_salary, 'employment_type': employment_type, 'years_of_employment': years_of_employment,
            'company_type': company_type, 'house_type': house_type, 'monthly_rent': monthly_rent,
            'family_size': family_size, 'dependents': dependents, 'school_fees': school_fees,
            'college_fees': college_fees, 'travel_expenses': travel_expenses, 'groceries_utilities': groceries_utilities,
            'other_monthly_expenses': other_monthly_expenses, 'existing_loans': existing_loans,
            'current_emi_amount': current_emi_amount, 'credit_score': credit_score, 'bank_balance': bank_balance,
            'emergency_fund': emergency_fund, 'emi_scenario': emi_scenario, 'requested_amount': requested_amount,
            'requested_tenure': requested_tenure
        }
        
        # Create DataFrame with all columns
        all_cols = ['age', 'gender', 'marital_status', 'education', 'monthly_salary',
                    'employment_type', 'years_of_employment', 'company_type', 'house_type',
                    'monthly_rent', 'family_size', 'dependents', 'school_fees',
                    'college_fees', 'travel_expenses', 'groceries_utilities',
                    'other_monthly_expenses', 'existing_loans', 'current_emi_amount',
                    'credit_score', 'bank_balance', 'emergency_fund', 'emi_scenario',
                    'requested_amount', 'requested_tenure']
        
        features_df = pd.DataFrame(data, index=[0])
        # Add any missing columns that the model expects
        for col in all_cols:
            if col not in features_df.columns:
                features_df[col] = 0
        
        with st.spinner("Analyzing profile and running models..."):
            try:
                # Apply the same feature engineering
                input_df_engineered = feature_engineer(features_df.copy())
                
                # --- START OF FIX ---
                
                # Define the label map exactly as in train.py
                label_map_reverse = {0: 'Eligible', 1: 'High_Risk', 2: 'Not_Eligible'}
                
                # Classification Prediction (will be a number, e.g., 0, 1, or 2)
                prediction_numeric = model_class.predict(input_df_engineered)[0]
                
                # Map the number back to a string label
                # Check if the prediction is a number first, otherwise just use it
                if isinstance(prediction_numeric, (int, np.integer)):
                    prediction_class = label_map_reverse.get(prediction_numeric, "Not_Eligible")
                else:
                    prediction_class = prediction_numeric # Use the raw prediction if it's already a string
                
                # --- END OF FIX ---

                prediction_class_proba = model_class.predict_proba(input_df_engineered)[0]
                
                # Regression Prediction
                prediction_reg = model_reg.predict(input_df_engineered)[0]
                
                st.header("Assessment Results")
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                # --- Classification Result ---
                with col1:
                    st.markdown("#### EMI Eligibility Assessment")
                    
                    # This logic will now work correctly
                    if prediction_class == "Eligible":
                        st.success(f"**Status: {prediction_class}**")
                        st.write("This applicant has a low-risk profile and strong affordability.")
                    elif prediction_class == "High_Risk":
                        st.warning(f"**Status: {prediction_class}**")
                        st.write("This applicant is a marginal case. Manual review or higher interest rates are recommended.")
                    else: # Not_Eligible
                        st.error(f"**Status: {prediction_class}**")
                        st.write("This applicant has a high-risk profile. Loan is not recommended.")
                    
                    st.markdown("Prediction Confidence:")
                    # Ensure probability labels match the map
                    proba_labels = [label_map_reverse.get(i, f"Unknown_{i}") for i in model_class.classes_]
                    proba_df = pd.DataFrame(prediction_class_proba, index=proba_labels, columns=['Confidence'])
                    st.dataframe(proba_df.style.format("{:.1%}"))

                # --- Regression Result ---
                with col2:
                    st.markdown("#### Recommended Max EMI")
                    max_emi = max(500, prediction_reg) # Ensure EMI is at least 500
                    st.info(f"**Recommended Max. Monthly EMI: ₹{max_emi:,.0f}**")
                    st.write("This is the maximum sustainable EMI amount predicted for this applicant's financial profile.")
                    
                    percent_of_salary = (max_emi / monthly_salary) * 100
                    st.markdown(f"This represents **{percent_of_salary:.1f}%** of their monthly salary.")
                    if percent_of_salary > 40:
                            st.warning("Warning: Recommended EMI is a high percentage of monthly salary.")
                    st.progress(int(percent_of_salary))

                # --- Engineered Features Display ---
                st.subheader("Engineered Financial Ratios")
                st.markdown("These are the key metrics calculated from the inputs, which heavily influence the prediction.")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Disposable Income (Monthly)", f"₹{input_df_engineered['disposable_income'].values[0]:,.0f}")
                c2.metric("Debt-to-Income (DTI) Ratio", f"{input_df_engineered['dti_ratio'].values[0]:.2f}")
                c3.metric("Savings-to-Income Ratio", f"{input_df_engineered['savings_to_income_ratio'].values[0]:.2f}")
                c4.metric("Financial Risk Score", f"{input_df_engineered['financial_risk_score'].values[0]:.2f}")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

    else:
        st.info("Please enter the financial details in the sidebar and click 'Assess Financial Risk'.")