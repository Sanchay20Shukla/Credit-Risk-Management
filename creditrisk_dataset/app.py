# 1 Good (lower Risk ) 0 Bad (higher Risk )
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


model = joblib.load("extra_trees_model.pkl")
encoders = {col : joblib.load(f"{col}_encoder.pkl") for col in ["Sex", "Housing", "Saving accounts", "Checking account","Purpose"]}

st.title("Credit Risk Prediction")
st.write("Enter applicant information to predict if the credit risk is good or bad")

age = st.number_input("Age", min_value = 18, max_value = 100, value = 30)
sex = st.selectbox("Sex", ["male", "female"])
job = st.number_input("Job(0-3)", min_value = 0, max_value = 3, value = 1)
housing = st.selectbox("Housing", ["own", "rent", "free"])
saving_accounts = st.selectbox("Saving accounts", ["little", "moderate", "rich","quite rich"])
checking_account = st.selectbox("Checking account", ["little", "moderate", "rich"])
credit_amount = st.number_input("Credit amount", min_value = 0, value = 5000)
duration = st.number_input("Duration(months)", min_value = 1, value = 12)

input_df = pd.DataFrame({
    "Age" : [age],
    "Sex" : [encoders["Sex"].transform([sex])[0]],
    "Job" : [job],
    "Housing" : [encoders["Housing"].transform([housing])[0]],
    "Saving accounts" : [encoders["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account" : [encoders["Checking account"].transform([checking_account])[0]],
    "Credit amount" : [credit_amount],
    "Duration" : [duration]
    })

if st.button("Predict Risk"):
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.success("The predicted credit risk is : **Good**")
    else:
        st.error("The predicted credit risk is : **Bad**")