
import streamlit as st
import pandas as pd
import numpy as np
import joblib

scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
kmeans = joblib.load("kmeans.pkl")
priority_map = joblib.load("priority_map.pkl")

def get_priority(df):
    X = pd.DataFrame({
        "Income": np.log1p(df["Income"]),
        "Total_Spent": np.log1p(df["Total_Spent"]),
        "Children": df["Children"],
        "Total_Purchases": np.log1p(df["Total_Purchases"])})
    return [priority_map[c] for c in kmeans.predict(pca.transform(scaler.transform(X)))]

st.title("🎯 Customer Priority Prediction")

income = st.number_input("Yearly Income", 0, 200000, 50000, step=1000)
total_spent = st.number_input("Total Spent", 0, 3000, 600)
children = st.number_input("Number of Children", 0, 5, 1)
total_purchases = st.number_input("Total Purchases", 0, 50, 15)

if st.button("Predict Priority", type="primary"):
    person = pd.DataFrame([{"Income": income, "Total_Spent": total_spent,
                            "Children": children, "Total_Purchases": total_purchases}])
    p = get_priority(person)[0]
    if p == "High Priority":
        st.success(f"🟢 {p}: Target first with premium offers")
    elif p == "Medium Priority":
        st.warning(f"🟠 {p}: Target with regular campaigns")
    else:
        st.error(f"🔴 {p}: Discount offers only")