import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="Medical Insurance Premium Predictor", layout="centered")

# Load Model
@st.cache_resource
def load_model():
    with open('xgb_10_features_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

try:
    artifact = load_model()
    model = artifact['model']
    features = artifact['features']
    metrics = artifact['metrics']
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# Header
st.title("🏥 Medical Insurance Premium Predictor (10 Features)")
st.write("Prediksi Biaya Premi Tahunan menggunakan model XGBoost dengan 10 Fitur Utama teratas.")

st.sidebar.header("Informasi Performa Model")
st.sidebar.write(f"**R² Score**: {metrics['r2']:.4f}")
st.sidebar.write(f"**MAE**: ${metrics['mae']:.2f}")
st.sidebar.write(f"**Akurasi**: {metrics['accuracy_pct']:.2f}%")

# Form Input Fitur
st.subheader("Input Data Pasien")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        annual_medical_cost = st.number_input("Biaya Medis Tahunan ($)", min_value=0.0, value=3000.0, step=100.0)
        total_claims_paid = st.number_input("Total Klaim yang Dibayar ($)", min_value=0.0, value=1000.0, step=100.0)
        avg_claim_amount = st.number_input("Rata-rata Jumlah Per Klaim ($)", min_value=0.0, value=500.0, step=50.0)
        risk_score = st.slider("Skor Risiko Kesehatan", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        chronic_count = st.number_input("Jumlah Penyakit Kronis", min_value=0, max_value=20, value=1, step=1)

    with col2:
        visits_last_year = st.number_input("Jumlah Kunjungan Medis (Tahun Lalu)", min_value=0, max_value=100, value=2, step=1)
        age = st.number_input("Usia (Tahun)", min_value=18, max_value=120, value=40, step=1)
        bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        deductible = st.number_input("Deduktibel ($)", min_value=0.0, value=1000.0, step=100.0)
        copay = st.number_input("Kopay / Uang Berobat ($)", min_value=0.0, value=20.0, step=5.0)

    submit_button = st.form_submit_button("Prediksi Premi Sekarang")

if submit_button:
    # Buat DataFrame input data sesuai susunan fitur yang digunakan saat pelatihan
    input_data = pd.DataFrame([[
        annual_medical_cost,
        total_claims_paid,
        avg_claim_amount,
        risk_score,
        chronic_count,
        visits_last_year,
        age,
        bmi,
        deductible,
        copay
    ]], columns=features)

    # Prediksi premi
    prediction = model.predict(input_data)[0]

    st.success(f"### 💰 Estimasi Premi Tahunan: **${prediction:,.2f}**")
