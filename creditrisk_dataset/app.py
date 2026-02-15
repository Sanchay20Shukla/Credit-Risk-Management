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
    st.title("Credit Risk")
    st.markdown("---")
    st.markdown("### Model Details")
    st.write("Algorithm: **Extra Trees**")
    st.write("Accuracy: **~82%**")
    st.markdown("---")
    st.caption("v1.0.2")

# Main Content Header
st.title("Credit Risk Assessment")

# Input Form
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        sex = st.selectbox("Sex", ["male", "female"])
        job = st.slider("Job Level", min_value=0, max_value=3, value=1)
        purpose = st.selectbox("Purpose", ["business", "car", "domestic appliances", "education", "furniture/equipment", "radio/TV", "repairs", "vacation/others"])

    with col2:
        housing = st.selectbox("Housing", ["own", "rent", "free"])
        saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "rich", "quite rich"])
        checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich"])
        credit_amount = st.number_input("Credit Amount", min_value=0, value=5000)
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

if st.button("Analyze Risk", use_container_width=True, type="primary"):
    pred = model.predict(input_df)[0]
    st.markdown("---")
    
    if pred == 1:
        st.success("Result: **GOOD RISK**")
    else:
        st.error("Result: **BAD RISK**")
