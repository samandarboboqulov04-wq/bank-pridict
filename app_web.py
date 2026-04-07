import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Bank Mijoz Bashorati",
    page_icon="🏦",
    layout="centered"
)

# Model yuklash
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "model.pkl"
    return joblib.load(model_path)

model = load_model()

# Title
st.title("🏦 Bank Mijoz Bashorat Tizimi")
st.markdown("---")

# Inputlar
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Mijoz yoshi", 18, 100, 30)
    qarzlar_soni = st.number_input("Qaramlar soni", 0, 100, 0)
    bank_bilan_oylar = st.number_input("Bank bilan oylar", 0, 120, 12)
    bank_mahsulotlari_soni = st.number_input("Mahsulotlar soni", 1, 20, 1)
    nofoal_oylar = st.number_input("Nofaol oylar (12 oy)", 0, 12, 0)

with col2:
    aloqa_soni = st.number_input("Aloqa soni (12 oy)", 0, 100, 0)
    kredit_limiti = st.number_input("Kredit limiti", 0.0, 100000.0, 5000.0)
    kredit_balansi = st.number_input("Kredit balansi", 0.0, 100000.0, 0.0)
    mavjud_kredit_qismi = st.number_input("Kredit ulushi", 0.0, 100000.0, 0.0)
    total_trans_amt = st.number_input("Tranzaksiya summasi", 0.0, 1000000.0, 1000.0)

st.markdown("---")

if st.button("🔮 Bashorat qilish"):
    try:
        # ✅ Faqat model kutgan feature'lar
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

        # ✅ Feature tartibi (model bilan 1:1 mos)
        df = df[[
            'mijoz_yoshi', 'qaramlar_soni', 'bank_bilan_oylar',
            'bank_mahsulotlari_soni', 'nofaol_oylar_12oy', 'aloqa_soni_12oy',
            'kredit_limiti', 'kredit_balansi', 'mavjud_kredit_qismi',
            'Total_Trans_Amt'
        ]]

        # Prediction (bu regression — target = kredit_foydalanish_darajasi)
        result = model.predict(df)[0]

        # Natijani chiroyli chiqarish
        st.success(f"📊 Xizmatdan foydalanish darajasi: {result:.4f}")

        # Agar foiz sifatida ko‘rsatmoqchi bo‘lsang:
        st.info(f"📈 Foizda: {result * 100:.2f}%")

    except Exception as e:
        st.error(f"❌ Xatolik: {e}")