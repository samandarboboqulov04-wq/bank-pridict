import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Bank Mijozlarining Xizmatdan foydalanish bashorati",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load model
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "model (2).pkl"
    return joblib.load(model_path)

model = load_model()

# Custom CSS
st.markdown("""
    <style>
    .main {
        max-width: 700px;
        margin: 0 auto;
    }
    .stTitle {
        text-align: center;
        color: #2d6cdf;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🏦 Bank Mijozlarining Xizmatdan foydalanish bashorati Model")
st.markdown("---")

# Instructions
st.markdown("""
Enter customer information below to predict their characteristics using our ML model.
""")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Mijoz yoshi", min_value=18, max_value=100, value=30)
    qarzlar_soni = st.number_input("Qarzlar soni", min_value=0, max_value=20, value=0)
    bank_bilan_oylar = st.number_input("Bank bilan ishlagan oylar", min_value=0, max_value=120, value=12)
    bank_mahsulotlari_soni = st.number_input("Bank mahsulotlari soni", min_value=1, max_value=10, value=1)
    nofoal_oylar = st.number_input("Faol bo‘lmagan oylar (12 oy ichida)", min_value=0, max_value=12, value=0)

with col2:
    aloqa_soni = st.number_input("Bog‘lanishlar soni (12 oy ichida)", min_value=0, max_value=20, value=0)
    kredit_limiti = st.number_input("Kredit limiti", min_value=0.0, value=5000.0, step=100.0)
    kredit_balansi = st.number_input("Kredit qoldig‘i", min_value=0.0, value=0.0, step=100.0)
    mavjud_kredit_qismi = st.number_input("Kredit ulushi (%)", min_value=0.0, max_value=100.0, value=0.0)
    total_trans_amt = st.number_input("Umumiy tranzaksiya summasi", min_value=0.0, value=1000.0, step=100.0)

# Prediction button
st.markdown("---")
col_button = st.columns(3)

with col_button[1]:
    if st.button("🔮 Make Prediction", use_container_width=True, type="primary"):
        try:
            # ✅ MUHIM: DataFrame ishlatamiz (TO‘G‘RI USUL)
            data = {
                "mijoz_yoshi": age,
                "qaramlar_soni": qarzlar_soni,
                "bank_bilan_oylar": bank_bilan_oylar,
                "bank_mahsulotlari_soni": bank_mahsulotlari_soni,
                "nofaol_oylar_12oy": nofoal_oylar,
                "aloqa_soni_12oy": aloqa_soni,
                "kredit_limiti": kredit_limiti,
                "kredit_balansi": kredit_balansi,
                "mavjud_kredit_qismi": mavjud_kredit_qismi,
                "Total_Trans_Amt": total_trans_amt
            }

            df = pd.DataFrame([data])

            # Prediction
            result = model.predict(df)[0]

            # Agar probability bo‘lsa (classification model bo‘lsa)
            try:
                prob = model.predict_proba(df)[0][1] * 100
                st.success(f"✅ Prediction Result: **{result:.4f}**")
                st.info(f"📊 Ehtimollik: **{prob:.2f}%**")
            except:
                st.success(f"✅ Prediction Result: **{result:.4f}**")

            # Debug (xohlasang ochib ko‘r)
            # st.write(df)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px;'>
    <p>Bank Customer Prediction System | Powered by Streamlit & Scikit-learn</p>
</div>
""", unsafe_allow_html=True)