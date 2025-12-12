import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Base path (file is placed in src/)
BASE_DIR = Path(__file__).resolve().parent

# File paths (all files are expected to be in the same folder as this script)
FILE_RAW = BASE_DIR / "Data-Balita-posyandu.csv"
FILE_ENCODED = BASE_DIR / "dataset_encoded.csv"
FILE_Z_COWOK = BASE_DIR / "ZScore_Balita_Laki-Laki.csv"
FILE_Z_CEWEK = BASE_DIR / "ZScore_Balita_perempuan.csv"
FILE_LABEL = BASE_DIR / "dataset+label.csv"
MODEL_PATH = BASE_DIR / "model_rf_smote.pkl"

st.set_page_config(page_title="Sistem Klasifikasi Gizi Balita", layout="wide")

st.title("Sistem Klasifikasi Gizi Balita — Streamlit")

menu = st.sidebar.selectbox("Menu", ["Klasifikasi", "Proses (tampilkan file & training)", "Tentang"])

# Utility to load CSV with graceful error message
@st.cache_data
def load_csv(path: Path):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        return None

# ---------- PAGE: Klasifikasi ----------
if menu == "Klasifikasi":
    st.header("Menu Klasifikasi — Input manual")
    st.markdown("Silakan masukkan nilai fitur kemudian klik **Prediksi**.")

    with st.form("form_predict"):
        col1, col2, col3 = st.columns(3)
        with col1:
            jenis_kelamin = st.selectbox("Jenis kelamin", options=[0,1], format_func=lambda x: "Perempuan (0)" if x==0 else "Laki-laki (1)")
            U = st.number_input("U (Usia dalam bulan)", min_value=0, max_value=216, value=24)
            BB = st.number_input("BB (Berat badan, kg)", min_value=0.0, value=8.0, step=0.1)
        with col2:
            TB = st.number_input("TB (Tinggi/length, cm)", min_value=0.0, value=63.0, step=0.1)
            LK = st.number_input("LK (Lingkar kepala, cm)", min_value=0.0, value=47.0, step=0.1)
            LILA = st.number_input("LILA (Lingkar lengan atas, cm)", min_value=0.0, value=12.0, step=0.1)
        with col3:
            vitaminA = st.selectbox("Vitamin A", options=[0,1,2], format_func=lambda x: {0:'B',1:'M',2:'T'}[x])
            asi = st.selectbox("ASI (Pemberian ASI)?", options=[0,1], format_func=lambda x: "Tidak (0)" if x==0 else "Ya (1)")
            lama_asi = st.number_input("Lama Pemberian ASI (Bulan)", min_value=0, max_value=120, value=3)

        submit = st.form_submit_button("Cek Status Gizi")

    if submit:
        # Load model
        if not MODEL_PATH.exists():
            st.error(f"Model tidak ditemukan di {MODEL_PATH}. Silakan letakkan model_rf_smote.pkl di folder src atau retrain di menu 'Proses'.")
        else:
            try:
                with open(MODEL_PATH, 'rb') as f:
                    model = pickle.load(f)
            except Exception as e:
                st.error(f"Gagal memuat model: {e}")
            else:
                # Build dataframe for prediction
                data_in = pd.DataFrame([{
                    "Jenis kelamin": int(jenis_kelamin),
                    "U": float(U),
                    "BB": float(BB),
                    "TB": float(TB),
                    "LK": float(LK),
                    "LILA": float(LILA),
                    "Vitamin A": int(vitaminA),
                    "Asi": int(asi),
                    "Lama Pemberian ASI (Bulan)": int(lama_asi)
                }])

                # If model expects encoded categorical values, we assume user uses same encoding as training.
                try:
                    pred = model.predict(data_in)[0]
                    st.success(f"Hasil Prediksi Status Gizi: {pred}")
                except Exception as e:
                    st.error("Gagal melakukan prediksi — kemungkinan model memerlukan preprocessing tambahan (scaler/encoder).")
                    st.exception(e)

# ---------- PAGE: Proses (tampilkan file & training) ----------
elif menu == "Proses (tampilkan file & training)":
    st.header("Halaman Proses — Tampilkan file dan (opsional) retrain")

    st.subheader("1) Hasil Preprocessing — Data-Balita-posyandu.csv")
    df_raw = load_csv(FILE_RAW)
    if df_raw is None:
        st.error(f"File tidak ditemukan: {FILE_RAW.name} — letakkan file di folder src.")
    else:
        st.dataframe(df_raw)

    st.subheader("2) Hasil Encoding — dataset_encoded.csv")
    df_encoded = load_csv(FILE_ENCODED)
    if df_encoded is None:
        st.error(f"File tidak ditemukan: {FILE_ENCODED.name}")
    else:
        st.dataframe(df_encoded)

    st.subheader("3) Hasil Z-score (Cowok) — ZScore_Balita_Laki-Laki.csv")
    df_z_cowok = load_csv(FILE_Z_COWOK)
    if df_z_cowok is None:
        st.error(f"File tidak ditemukan: {FILE_Z_COWOK.name}")
    else:
        st.dataframe(df_z_cowok)

    st.subheader("4) Hasil Z-score (Cewek) — ZScore_Balita_perempuan.csv")
    df_z_cewek = load_csv(FILE_Z_CEWEK)
    if df_z_cewek is None:
        st.error(f"File tidak ditemukan: {FILE_Z_CEWEK.name}")
    else:
        st.dataframe(df_z_cewek)

    st.subheader("5) Hasil Majority Voting — dataset+label.csv")
    df_label = load_csv(FILE_LABEL)
    if df_label is None:
        st.error(f"File tidak ditemukan: {FILE_LABEL.name}")
    else:
        st.dataframe(df_label)
        st.markdown("**Jumlah tiap label majority voting:**")
        vc = df_label["Status_Gizi_Mayority"].value_counts()
        st.write(vc)

    st.markdown("---")
    st.subheader("6) Pelatihan Model (SMOTE + RandomForest) — jalankan jika ingin retrain dan menyimpan model")
    st.write("Fitur yang digunakan: ['Jenis kelamin', 'U', 'BB', 'TB', 'LK', 'LILA', 'Vitamin A', 'Asi', 'Lama Pemberian ASI (Bulan)']")

    if df_label is None:
        st.info("Tidak ada file dataset+label.csv — tidak bisa melatih model.")
    else:
        if st.button("Mulai Training (SMOTE + RandomForest)"):
            with st.spinner("Training... (akan menyimpan model ke src/model_rf_smote.pkl jika berhasil) "):
                try:
                    df = df_label.copy()
                    fitur = ["Jenis kelamin", "U", "BB", "TB", "LK", "LILA", "Vitamin A", "Asi", "Lama Pemberian ASI (Bulan)"]
                    # Handle small differences in column naming
                    for col in fitur:
                        if col not in df.columns:
                            st.warning(f"Kolom fitur tidak ditemukan di dataset: {col}")

                    X = df[fitur].copy()
                    y = df["Status_Gizi_Mayority"].copy()

                    # Encode categorical columns if necessary
                    encoders = {}
                    for c in X.select_dtypes(include=['object', 'category']).columns:
                        le = LabelEncoder()
                        X[c] = le.fit_transform(X[c].astype(str))
                        encoders[c] = le

                    # Train-test split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

                    st.write("Distribusi sebelum SMOTE:")
                    st.write(y_train.value_counts())

                    # SMOTE with safe sampling strategy from user
                    smote = SMOTE(sampling_strategy={
                        'Gizi Over': 400,
                        'Gizi Kurang': 400
                    })
                    X_res, y_res = smote.fit_resample(X_train, y_train)

                    st.write("Distribusi setelah SMOTE:")
                    st.write(y_res.value_counts())

                    # Train RandomForest with chosen hyperparameters
                    rf_best = RandomForestClassifier(criterion='entropy', max_depth=20, max_features='sqrt', n_estimators=100, random_state=42)
                    rf_best.fit(X_res, y_res)

                    # Evaluate
                    y_pred = rf_best.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
                    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

                    st.success("Training selesai")
                    st.write({"accuracy": acc, "precision_macro": prec, "recall_macro": rec, "f1_macro": f1})
                    st.text("Classification Report:\n" + classification_report(y_test, y_pred, zero_division=0))

                    # Save model (and encoders) as a dict
                    save_obj = {
                        'model': rf_best,
                        'encoders': encoders
                    }
                    with open(MODEL_PATH, 'wb') as f:
                        pickle.dump(save_obj, f)

                    st.success(f"Model disimpan ke: {MODEL_PATH.name}")

                except Exception as e:
                    st.exception(e)
