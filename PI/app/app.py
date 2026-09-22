import streamlit as st  # type: ignore[import-not-found]
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Estimasi Harga Rumah · Jabodetabek",
    page_icon="🏡",
    layout="centered",
)

# =========================================================
# DESAIN — TOKEN & GAYA
# =========================================================
# Palet terinspirasi dari material rumah: genteng tanah liat (clay),
# tanah/pekarangan (deep green), dan kertas sertifikat tanah (ivory).
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    --clay:        #B5502A;
    --clay-dark:   #8F3E20;
    --land:        #2F4A3C;
    --land-light:  #4B6B57;
    --paper:       #F2EDE2;
    --card:        #FFFDF8;
    --ink:         #211D18;
    --ink-soft:    #6B6355;
    --gold:        #A9782F;
    --line:        #E4DCC8;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: var(--ink);
}

.stApp {
    background: var(--paper);
}

/* ---------- Hero ---------- */
.hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--clay);
    margin-bottom: 0.4rem;
}
.hero-title {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 2.5rem;
    line-height: 1.15;
    color: var(--ink);
    margin-bottom: 0.6rem;
}
.hero-title em {
    font-style: italic;
    color: var(--land);
}
.hero-sub {
    font-size: 1rem;
    color: var(--ink-soft);
    max-width: 34rem;
    line-height: 1.55;
}

/* Signature divider: pola geometris kecil, terinspirasi motif ubin tradisional */
.motif-divider {
    height: 10px;
    margin: 1.6rem 0 2rem 0;
    opacity: 0.55;
    background-image:
        linear-gradient(45deg, var(--clay) 25%, transparent 25%),
        linear-gradient(-45deg, var(--clay) 25%, transparent 25%);
    background-size: 14px 14px;
    background-position: 0 0;
    border-radius: 2px;
}

/* ---------- Section labels ---------- */
.section-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--land);
    margin: 0 0 0.35rem 0;
}
.section-title {
    font-family: 'Fraunces', serif;
    font-size: 1.35rem;
    font-weight: 600;
    color: var(--ink);
    margin-bottom: 0.9rem;
}

/* ---------- Card wrapper ---------- */
.card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1.6rem 1.7rem 1.2rem 1.7rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 1px 2px rgba(33, 29, 24, 0.04);
}

/* ---------- Inputs ---------- */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--ink-soft);
}
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] div[data-baseweb="select"] {
    border-radius: 9px !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: var(--clay) !important;
    box-shadow: 0 0 0 1px var(--clay) !important;
}

/* ---------- Button ---------- */
div[data-testid="stButton"] button {
    background: var(--clay);
    color: #FFF9F2;
    font-weight: 600;
    font-size: 1rem;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    transition: transform 0.12s ease, box-shadow 0.12s ease, background 0.12s ease;
    box-shadow: 0 2px 0 var(--clay-dark);
}
div[data-testid="stButton"] button:hover {
    background: var(--clay-dark);
    transform: translateY(-1px);
    box-shadow: 0 3px 0 var(--clay-dark);
    color: #FFF9F2;
}
div[data-testid="stButton"] button:active {
    transform: translateY(1px);
    box-shadow: 0 1px 0 var(--clay-dark);
}

/* ---------- Result panel ---------- */
.result-card {
    background: var(--land);
    border-radius: 16px;
    padding: 1.8rem 1.8rem;
    margin: 0.4rem 0 1.6rem 0;
    color: #F4F1E8;
}
.result-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #C9D6CC;
    margin-bottom: 0.5rem;
}
.result-value {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 2.3rem;
    color: #F4F1E8;
    letter-spacing: -0.01em;
}
.result-meta {
    font-size: 0.85rem;
    color: #B9C6BC;
    margin-top: 0.6rem;
}

/* ---------- Footer ---------- */
.footer-note {
    font-size: 0.8rem;
    color: var(--ink-soft);
    text-align: center;
    margin-top: 1.2rem;
}

/* Sembunyikan elemen bawaan Streamlit yang tidak perlu */
#MainMenu, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL, FITUR, DAN SCALER
# =========================================================
@st.cache_resource
def load_artifacts():
    model_dir = Path(__file__).resolve().parent
    model = joblib.load(model_dir / "random_forest_model.pkl")
    fitur_kolom = joblib.load(model_dir / "fitur_model.pkl")
    scaler = joblib.load(model_dir / "scaler.pkl")
    return model, fitur_kolom, scaler

try:
    model, fitur_kolom, scaler = load_artifacts()
except FileNotFoundError:
    st.error(
        "File model tidak ditemukan. Pastikan `random_forest_model.pkl`, "
        "`fitur_model.pkl`, dan `scaler.pkl` berada di folder yang sama "
        "dengan app.py ini (hasil dari notebook training)."
    )
    st.stop()

daftar_kota_dummy = sorted([
    col.replace("city_", "") for col in fitur_kolom if col.startswith("city_")
])
# Kota baseline (di-drop saat get_dummies drop_first=True) tidak punya kolom
# dummy sendiri, jadi isi manual di sini kalau kamu tahu namanya.
KOTA_BASELINE = None  # contoh: "Jakarta Barat"

daftar_kota = daftar_kota_dummy.copy()
if KOTA_BASELINE and KOTA_BASELINE not in daftar_kota:
    daftar_kota = [KOTA_BASELINE] + daftar_kota

# =========================================================
# HERO
# =========================================================
st.markdown('<div class="hero-eyebrow">Estimasi Harga · Jabodetabek</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Berapa nilai rumahmu?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Isi detail tanah dan bangunan di bawah ini. '
    'Model Random Forest akan memperkirakan harga pasar berdasarkan data '
    'properti di kawasan Jabodetabek.</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="motif-divider"></div>', unsafe_allow_html=True)

# =========================================================
# FORM INPUT
# =========================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-eyebrow">Langkah 1</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Detail Properti</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    land_size_m2 = st.number_input(
        "Luas Tanah (m²)", min_value=0.0, max_value=10000.0, value=100.0, step=5.0
    )
    bedrooms = st.number_input(
        "Jumlah Kamar Tidur", min_value=0, max_value=20, value=3, step=1
    )
    floors = st.number_input(
        "Jumlah Lantai", min_value=0, max_value=10, value=1, step=1
    )

with col2:
    building_size_m2 = st.number_input(
        "Luas Bangunan (m²)", min_value=0.0, max_value=10000.0, value=80.0, step=5.0
    )
    bathrooms = st.number_input(
        "Jumlah Kamar Mandi", min_value=0, max_value=20, value=2, step=1
    )
    garages = st.number_input(
        "Jumlah Garasi", min_value=0, max_value=10, value=1, step=1
    )

city = st.selectbox("Kota", options=daftar_kota)
st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# FUNGSI PREDIKSI
# =========================================================
def prediksi_harga(land_size_m2, building_size_m2, bedrooms, bathrooms,
                    floors, garages, city):
    data_baru = pd.DataFrame(
        np.zeros((1, len(fitur_kolom))), columns=fitur_kolom
    )

    data_baru.loc[0, "land_size_m2"] = land_size_m2
    data_baru.loc[0, "building_size_m2"] = building_size_m2
    data_baru.loc[0, "bedrooms"] = bedrooms
    data_baru.loc[0, "bathrooms"] = bathrooms
    data_baru.loc[0, "floors"] = floors
    data_baru.loc[0, "garages"] = garages

    kolom_city = f"city_{city}"
    if kolom_city in data_baru.columns:
        data_baru.loc[0, kolom_city] = 1
    # Jika city adalah kategori baseline (di-drop saat training),
    # semua kolom city_* tetap 0 -> ini sudah benar/wajar.

    data_scaled = scaler.transform(data_baru)
    harga = model.predict(data_scaled)[0]
    return harga

# =========================================================
# TOMBOL & HASIL
# =========================================================
st.markdown('<div class="section-eyebrow" style="margin-top:0.2rem;">Langkah 2</div>', unsafe_allow_html=True)
prediksi_diminta = st.button("Hitung Estimasi Harga", use_container_width=True)

if prediksi_diminta:
    if land_size_m2 == 0 or building_size_m2 == 0:
        st.warning("Luas tanah dan luas bangunan sebaiknya lebih besar dari 0 m².")
    if building_size_m2 > land_size_m2 * 5 and land_size_m2 > 0:
        st.warning("Luas bangunan jauh lebih besar dari luas tanah — cek kembali inputnya.")

    harga_prediksi = prediksi_harga(
        land_size_m2, building_size_m2, bedrooms, bathrooms,
        floors, garages, city
    )
    harga_per_m2 = harga_prediksi / building_size_m2 if building_size_m2 > 0 else 0

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Estimasi Harga Jual</div>
        <div class="result-value">Rp {harga_prediksi:,.0f}</div>
        <div class="result-meta">≈ Rp {harga_per_m2:,.0f} / m² bangunan · {city}</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Lihat detail input"):
        st.write({
            "Luas Tanah (m²)": land_size_m2,
            "Luas Bangunan (m²)": building_size_m2,
            "Kamar Tidur": bedrooms,
            "Kamar Mandi": bathrooms,
            "Lantai": floors,
            "Garasi": garages,
            "Kota": city,
        })

st.markdown(
    '<div class="footer-note">Model: Random Forest Regressor · '
    'Fitur: luas tanah, luas bangunan, kamar tidur, kamar mandi, lantai, garasi, kota.</div>',
    unsafe_allow_html=True,
)