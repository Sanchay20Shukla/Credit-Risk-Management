# 1 Good (lower Risk ) 0 Bad (higher Risk )
import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load model and encoders from the same folder as this script (required for Streamlit Cloud)
APP_DIR = os.path.dirname(os.path.abspath(__file__))

def _pkl_path(name):
    path = os.path.join(APP_DIR, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path} (app dir: {APP_DIR})")
    return path

model = joblib.load(_pkl_path("extra_trees_model.pkl"))
encoder_cols = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]
encoders = {col: joblib.load(_pkl_path(f"{col}_encoder.pkl")) for col in encoder_cols}

# Set page configuration for a professional B2B look
st.set_page_config(
    page_title="Credit Risk Analyzer | Business Edition",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and encoders from the same folder as this script (required for Streamlit Cloud)
APP_DIR = os.path.dirname(os.path.abspath(__file__))

def _pkl_path(name):
    path = os.path.join(APP_DIR, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path} (app dir: {APP_DIR})")
    return path

model = joblib.load(_pkl_path("extra_trees_model.pkl"))
encoder_cols = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]
encoders = {col: joblib.load(_pkl_path(f"{col}_encoder.pkl")) for col in encoder_cols}

# Sidebar Content
with st.sidebar:
    st.image("https://img.icons8.com/external-flat-icons-inorganic-blue/100/external-risk-insurance-flat-icons-inorganic-blue.png", width=80)
    st.title("Credit Intelligence")
    st.markdown("---")
    st.info("""
    **About this tool:**
    This B2B dashboard utilizes an Extra Trees Classifier to evaluate creditworthiness for business and individual applicants.
    """)
    st.markdown("### 📊 Metrics")
    st.write("- Model: **Extra Trees**")
    st.write("- Accuracy: **82% (approx)**")
    st.write("- Data: **German Credit Dataset**")
    st.markdown("---")
    st.caption("© 2026 Credit Risk Management Solutions")

# Main Content Header
st.title("🛡️ Credit Risk Assessment Dashboard")
st.markdown("Welcome to the professional credit evaluation tool. Enter the applicant's data below for real-time risk analysis.")

# Input Form
with st.container(border=True):
    st.subheader("📋 Applicant Information")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Personal Information**")
        age = st.number_input("Age (Years)", min_value=18, max_value=100, value=30, help="Applicant's age in years.")
        sex = st.selectbox("Sex", ["male", "female"], help="Applicant's biological sex.")
        job = st.slider("Job Level (0-3)", min_value=0, max_value=3, value=1, help="0: unskilled/non-res, 1: unskilled/res, 2: skilled, 3: highly skilled")
        purpose = st.selectbox("Purpose of Loan", ["business", "car", "domestic appliances", "education", "furniture/equipment", "radio/TV", "repairs", "vacation/others"])

    with col2:
        st.write("**Financial & Housing**")
        housing = st.selectbox("Housing Status", ["own", "rent", "free"])
        saving_accounts = st.selectbox("Saving Accounts Status", ["little", "moderate", "rich", "quite rich"])
        checking_account = st.selectbox("Checking Account Status", ["little", "moderate", "rich"])
        credit_amount = st.number_input("Credit Amount (DM)", min_value=0, value=5000)
        duration = st.number_input("Duration (Months)", min_value=1, value=12)

# Encoding and Prediction
input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [encoders["Sex"].transform([sex])[0]],
    "Job": [job],
    "Housing": [encoders["Housing"].transform([housing])[0]],
    "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
    "Credit amount": [credit_amount],
    "Duration": [duration],
    "Purpose": [encoders["Purpose"].transform([purpose])[0]]
})

st.markdown("<br>", unsafe_allow_html=True) # Spacing

if st.button("🚀 Run Risk Analysis", use_container_width=True, type="primary"):
    with st.spinner("Analyzing applicant data..."):
        # We need a purposeful list for encoding if used, adding it back
        # The original code only used these 8 features, purpose was loaded but not used in input_df
        pred = model.predict(input_df)[0]
        
        st.markdown("---")
        st.subheader("🔍 Analysis Result")
        
        if pred == 1:
            st.success("### ✅ THE PREDICTED CREDIT RISK IS: **GOOD**")
            st.balloons()
            st.markdown("""
            **Summary Analysis:** 
            Based on the provided demographic and financial indicators, this applicant shows a statistically high probability of meeting their credit obligations. 
            The risk level is within acceptable corporate thresholds.
            """)
        else:
            st.error("### ⚠️ THE PREDICTED CREDIT RISK IS: **BAD**")
            st.markdown("""
            **Summary Analysis:** 
            The risk engine has identified several indicators suggesting a potential difficulty in credit fulfillment.
            Additional security or a thorough manual review is recommended before final approval.
            """)
            
    # Add a data visualization if you want a dashboard feel
    st.divider()
    st.caption("Probability analysis is based on historical patterns in the German Credit Dataset.")
