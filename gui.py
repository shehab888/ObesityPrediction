# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

gender_map = {'Male': 1, 'Female': 0}
family_history_map = {'no': 0, 'yes': 1}
favc_map = {'no': 0, 'yes': 1}
caec_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
scc_map = {'no': 0, 'yes': 1}
mtrans_map = {'Public_Transportation': 0, 'Walking': 1, 'Automobile': 2, 'Motorbike': 3, 'Bike': 4}
calc_map = caec_map  # نفس مابينج CAEC

def handle_outliers(df):
    for col in df.select_dtypes(include=np.number).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower, lower, np.where(df[col] > upper, upper, df[col]))
    return df

st.title("Obesity Prediction App")

#
gender = st.selectbox("Gender", list(gender_map.keys()))
age = st.number_input("Age", min_value=1, max_value=100)
height = st.number_input("Height (meters)", min_value=1.0, max_value=2.5)
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0)
family_history = st.selectbox("Family History with Overweight", list(family_history_map.keys()))
favc = st.selectbox("Do you eat high caloric food frequently?", list(favc_map.keys()))
fcvc = st.slider("Frequency of vegetables consumption (FCVC)", 1.0, 3.0, 2.0)
ncp = st.slider("Number of main meals (NCP)", 1.0, 4.0, 3.0)
caec = st.selectbox("Consumption of food between meals (CAEC)", list(caec_map.keys()))
ch2o = st.slider("Daily water intake (CH2O)", 1.0, 3.0, 2.0)
scc = st.selectbox("Do you monitor your calorie consumption (SCC)?", list(scc_map.keys()))
faf = st.slider("Physical activity frequency (FAF)", 0.0, 3.0, 1.0)
tue = st.slider("Time using technology devices (TUE)", 0.0, 3.0, 1.0)
calc = st.selectbox("Alcohol consumption (CALC)", list(calc_map.keys()))
mtrans = st.selectbox("Transportation method (MTRANS)", list(mtrans_map.keys()))

if st.button("Predict"):
    user_input = {
        'Gender': gender_map[gender],
        'Age': age,
        'Height': height,
        'Weight': weight,
        'family_history_with_overweight': family_history_map[family_history],
        'FAVC': favc_map[favc],
        'FCVC': fcvc,
        'NCP': ncp,
        'CAEC': caec_map[caec],
        'CH2O': ch2o,
        'SCC': scc_map[scc],
        'FAF': faf,
        'TUE': tue,
        'CALC': calc_map[calc],
        'MTRANS': mtrans_map[mtrans],

    }

    output_map = {
        0: "Insufficient_Weight",
        1: "Normal_Weight",
        2: "Overweight_Level_I",
        3: "Overweight_Level_II",
        4: "Obesity_Type_I",
        5: "Obesity_Type_II",
        6: "Obesity_Type_III"
    }

    df = pd.DataFrame([user_input])
    df = handle_outliers(df)
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)[0]
    proba = model.predict_proba(df_scaled)[0]

    st.success(f"Prediction: {output_map[prediction]}")
    # st.write(f"Probability: {proba}")
