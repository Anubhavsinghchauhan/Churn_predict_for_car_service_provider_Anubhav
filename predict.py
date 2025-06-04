import streamlit as st
import pickle

import pandas as pd
import numpy as np
# Custom page config
st.set_page_config(page_title="Driver Churn Predictor", page_icon="🚗", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
        font-family: 'Arial', sans-serif;
    }
    h1 {
        color: #1f77b4;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 10px;
    }
    .footer {
        position: fixed;
        bottom: 10px;
        width: 100%;
        text-align: center;
        color: grey;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
with open("xgmodel.pkl", "rb") as f:
    model = pickle.load(f)



# Load pre-encoded data
df = pd.read_csv("pre_encoded_data.csv")
city_target_mean = df.groupby("City")["churn"].mean()

# Title
st.title("🚗 Driver Churn Prediction App")
st.markdown("### Enter driver details below to predict churn risk:")

# Split layout into two columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=70)
    gender = st.selectbox("Gender", ["Male", "Female"])
    st.caption("**Education_Level**: 0 → 10+, 1 → 12+, 2 → Graduate")
    education_level = st.selectbox("Education Level", sorted(df["Education_Level"].unique()))
    income = st.number_input("Income")
    joining_designation = st.selectbox("Joining Designation", sorted(df["Joining Designation"].unique()))
    grade = st.selectbox("Grade", sorted(df["Grade"].unique()))

with col2:
    total_business_value = st.number_input("Total Business Value")
    st.caption("**Quarterly Rating**: 1 to 5 — higher is better")
    quarterly_rating = st.selectbox("Quarterly Rating", [1, 2, 3, 4, 5])
    quarterly_rating_increased = st.selectbox("Quarterly Rating Increased?", [0, 1])
    income_increased = st.selectbox("Income Increased?", [0, 1])
    doj_year = st.number_input("Year of Joining", min_value=2014, max_value=2022)
    doj_month = st.selectbox("Month of Joining", list(range(1, 13)))
    city = st.selectbox("City", sorted(df["City"].unique()))

# Encode city
city_encoded = city_target_mean.get(city, 0.0)

# Final input for model
input_data = np.array([[
    age,
    1 if gender == "Male" else 0,
    education_level,
    income,
    joining_designation,
    grade,
    total_business_value,
    quarterly_rating,
    quarterly_rating_increased,
    income_increased,
    doj_year,
    doj_month,
    city_encoded
]])

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error("⚠️ The driver is likely to **churn**.")
    else:
        st.success("✅ The driver is **not likely** to churn.")

# Footer
st.markdown('<div class="footer">Made with ❤️ by <b>Anubhav</b></div>', unsafe_allow_html=True)
