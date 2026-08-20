"""
app.py
======
Dashboard interactivo — Cyber Threat Predictor
Ejecutar: streamlit run src/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import joblib
import os

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Cyber Threat Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────
MODELS_DIR    = "models"
PROCESSED_DIR = os.path.join("data", "processed")
REPORTS_DIR   = os.path.join("reports", "figures")

FEATURE_COLS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]

CLASS_COLORS = {
    "normal": "#4CAF50",
    "DoS":    "#F44336",
    "Probe":  "#2196F3",
    "R2L":    "#FF9800",
    "U2R":    "#9C27B0",
}

CLASS_ICONS = {
    "normal": "✅",
    "DoS":    "💥",
    "Probe":  "🔍",
    "R2L":    "🔓",
    "U2R":    "⚠️",
}

CLASS_DESC = {
    "normal": "Tráfico legítimo de red. Sin amenaza detectada.",
    "DoS":    "Ataque de Denegación de Servicio. Intento de saturar recursos.",
    "Probe":  "Escaneo de red. Reconocimiento previo a un ataque.",
    "R2L":    "Acceso remoto no autorizado. Intento de intrusión externa.",
    "U2R":    "Escalada de privilegios. Intento de acceso root.",
}

# ─────────────────────────────────────────────
# CARGA DE MODELOS Y DATOS (cacheado)
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    rf  = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    xgb = joblib.load(os.path.join(MODELS_DIR, "xgboost.pkl"))
    lr  = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    le  = joblib.load(os.path.join(MODELS_DIR, "label_encoder_multi.pkl"))
    scaler   = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
    return rf, xgb, lr, le, scaler, encoders

RAW_DIR = os.path.join("data", "raw")

COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]

DOS_ATTACKS   = {"back","land","neptune","pod","smurf","teardrop","apache2",
                 "udpstorm","processtable","worm","mailbomb"}
PROBE_ATTACKS = {"ipsweep","nmap","portsweep","satan","mscan","saint"}
R2L_ATTACKS   = {"ftp_write","guess_passwd","imap","multihop","phf","spy",
                 "warezclient","warezmaster","sendmail","named","snmpgetattack",
                 "snmpguess","xlock","xsnoop","httptunnel"}
U2R_ATTACKS   = {"buffer_overflow","loadmodule","perl","rootkit","sqlattack",
                 "xterm","ps"}

def map_label(label):
    label = label.strip().lower().rstrip(".")
    if label == "normal":  return "normal"
    if label in DOS_ATTACKS:   return "DoS"
    if label in PROBE_ATTACKS: return "Probe"
    if label in R2L_ATTACKS:   return "R2L"
    if label in U2R_ATTACKS:   return "U2R"
    return "other"

@st.cache_data
def load_data():
    # Si ya existen los CSVs procesados, usarlos directamente
    train_path = os.path.join(PROCESSED_DIR, "train_processed.csv")
    test_path  = os.path.join(PROCESSED_DIR, "test_processed.csv")

    if os.path.exists(train_path) and os.path.exists(test_path):
        train = pd.read_csv(train_path)
        test  = pd.read_csv(test_path)
        return train, test

    # Si no existen, generar desde raw
    train = pd.read_csv(os.path.join(RAW_DIR, "KDDTrain+.txt"),
                        header=None, names=COLUMNS)
    test  = pd.read_csv(os.path.join(RAW_DIR, "KDDTest+.txt"),
                        header=None, names=COLUMNS)

    for df in [train, test]:
        df["label_multi"]  = df["label"].apply(map_label)
        df["label_binary"] = (df["label_multi"] != "normal").astype(int)
        for col in ["protocol_type", "service", "flag"]:
            df[col] = encoders[col].transform(
                df[col].apply(lambda x: x if x in encoders[col].classes_ else
                              encoders[col].classes_[0])
            )

    return train, test


# ─────────────────────────────────────────────
# CSS PERSONALIZADO
# ─────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1e3a5f, #0d2137);
    border: 1px solid #2196F3;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 5px 0;
}
.metric-value { font-size: 2rem; font-weight: bold; color: #64B5F6; }
.metric-label { font-size: 0.85rem; color: #90CAF9; margin-top: 4px; }

.threat-card {
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
    text-align: center;
}
.stSidebar { background-color: #0d1117; }
h1, h2, h3 { color: #64B5F6; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Cyber Threat Predictor")
    st.markdown("---")
    page = st.radio(
        "Navegación",
        ["🏠 Inicio", "📊 EDA & Datos", "🤖 Modelos", "🔮 Predictor", "📈 Comparativa"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Dataset:** NSL-KDD")
    st.markdown("**Modelos:** RF · XGBoost · LR")
    st.markdown("**Técnica:** SMOTE")
    st.markdown("---")
    st.markdown(
        "<small>Desarrollado por<br><b>Alejandro Gaspar Rivera</b></small>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# CARGA
# ─────────────────────────────────────────────
rf, xgb, lr, le, scaler, encoders = load_models()
train_df, test_df = load_data()

X_test = test_df[FEATURE_COLS].values
y_test_raw = test_df["label_multi"].apply(
    lambda x: x if x in le.classes_ else "normal"
)
y_test = le.transform(y_test_raw)
y_test_bin = test_df["label_binary"].values

# ─────────────────────────────────────────────
# PÁGINA: INICIO
# ─────────────────────────────────────────────
if page == "🏠 Inicio":
    st.title("🛡️ Cyber Threat Predictor")
    st.markdown("### Sistema de Clasificación de Amenazas de Red — NSL-KDD")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    total = len(train_df) + len(test_df)
    n_classes = len(le.classes_)

    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{total:,}</div>
            <div class="metric-label">Total Registros</div></div>""",
            unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{len(FEATURE_COLS)}</div>
            <div class="metric-label">Features</div></div>""",
            unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{n_classes}</div>
            <div class="metric-label">Clases</div></div>""",
            unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">3</div>
            <div class="metric-label">Modelos ML</div></div>""",
            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎯 Categorías de Amenaza")
    cols = st.columns(5)
    for i, (cls, icon) in enumerate(CLASS_ICONS.items()):
        with cols[i]:
            color = CLASS_COLORS[cls]
            st.markdown(f"""
            <div style="background:{color}22; border:2px solid {color};
                 border-radius:10px; padding:15px; text-align:center;">
                <div style="font-size:2rem">{icon}</div>
                <div style="font-weight:bold; color:{color}; font-size:1.1rem">{cls}</div>
                <div style="font-size:0.75rem; color:#ccc; margin-top:5px">
                    {CLASS_DESC[cls]}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏗️ Arquitectura del Pipeline")
    st.markdown("""
    ```
    KDDTrain+.txt ──┐
                    ├─► preprocess.py ─► SMOTE ─► train.py ─► models/
    KDDTest+.txt  ──┘                                              │
                                                                   ▼
                                                            app.py (Dashboard)
    ```
    """)

# ─────────────────────────────────────────────
# PÁGINA: EDA & DATOS
# ─────────────────────────────────────────────
elif page == "📊 EDA & Datos":
    st.title("📊 Análisis Exploratorio de Datos")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Distribución", "Features", "Muestra"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Distribución — Train")
            dist_train = train_df["label_multi"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = [CLASS_COLORS.get(c, "#999") for c in dist_train.index]
            ax.bar(dist_train.index, dist_train.values, color=colors)
            ax.set_facecolor("#0d1117")
            fig.patch.set_facecolor("#0d1117")
            ax.tick_params(colors="white")
            ax.set_title("Train", color="white")
            for i, v in enumerate(dist_train.values):
                ax.text(i, v + 500, f"{v:,}", ha="center", color="white", fontsize=8)
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("#### Distribución — Test")
            dist_test = test_df["label_multi"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = [CLASS_COLORS.get(c, "#999") for c in dist_test.index]
            ax.bar(dist_test.index, dist_test.values, color=colors)
            ax.set_facecolor("#0d1117")
            fig.patch.set_facecolor("#0d1117")
            ax.tick_params(colors="white")
            ax.set_title("Test", color="white")
            for i, v in enumerate(dist_test.values):
                ax.text(i, v + 100, f"{v:,}", ha="center", color="white", fontsize=8)
            st.pyplot(fig)
            plt.close()

        st.markdown("#### Comparativa Train vs Test")
        all_classes = sorted(set(train_df["label_multi"].unique()) |
                             set(test_df["label_multi"].unique()))
        comp = pd.DataFrame({
            "Clase":  all_classes,
            "Train":  [train_df["label_multi"].value_counts().get(c, 0) for c in all_classes],
            "Test":   [test_df["label_multi"].value_counts().get(c, 0)  for c in all_classes],
        })
        st.dataframe(comp.set_index("Clase"), use_container_width=True)

    with tab2:
        st.markdown("#### Top 15 Features por Varianza")
        img_path = os.path.join(REPORTS_DIR, "02_feature_variance.png")
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            variances = train_df[FEATURE_COLS].var().sort_values(ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(variances.index[::-1], variances.values[::-1], color="#2196F3")
            ax.set_facecolor("#0d1117"); fig.patch.set_facecolor("#0d1117")
            ax.tick_params(colors="white")
            st.pyplot(fig); plt.close()

        st.markdown("#### Feature Importance — Random Forest")
        img_path2 = os.path.join(REPORTS_DIR, "05_feature_importance.png")
        if os.path.exists(img_path2):
            st.image(img_path2, use_container_width=True)

    with tab3:
        st.markdown("#### Muestra del dataset procesado")
        n = st.slider("Número de filas", 5, 100, 20)
        st.dataframe(test_df[FEATURE_COLS + ["label_multi", "label_binary"]].head(n),
                     use_container_width=True)

# ─────────────────────────────────────────────
# PÁGINA: MODELOS
# ─────────────────────────────────────────────
elif page == "🤖 Modelos":
    st.title("🤖 Evaluación de Modelos")
    st.markdown("---")

    model_choice = st.selectbox(
        "Selecciona un modelo",
        ["Random Forest (Multiclase)", "XGBoost (Multiclase)", "Logistic Regression (Binario)"]
    )

    if model_choice == "Random Forest (Multiclase)":
        model, y_pred = rf, rf.predict(X_test)
        y_true = y_test
        class_names = list(le.classes_)
        is_binary = False
    elif model_choice == "XGBoost (Multiclase)":
        model, y_pred = xgb, xgb.predict(X_test)
        y_true = y_test
        class_names = list(le.classes_)
        is_binary = False
    else:
        model, y_pred = lr, lr.predict(X_test)
        y_true = y_test_bin
        class_names = ["Normal", "Ataque"]
        is_binary = True

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ Accuracy", f"{acc:.4f}")
    with col2:
        st.metric("🎯 F1-Score (weighted)", f"{f1:.4f}")
    with col3:
        st.metric("📦 Muestras test", f"{len(y_true):,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Matriz de Confusión")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_ylabel("Real", color="white")
        ax.set_xlabel("Predicho", color="white")
        ax.set_facecolor("#0d1117"); fig.patch.set_facecolor("#0d1117")
        ax.tick_params(colors="white")
        plt.xticks(rotation=45)
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown("#### Reporte de Clasificación")
        report = classification_report(y_true, y_pred,
                                       target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)

# ─────────────────────────────────────────────
# PÁGINA: PREDICTOR
# ─────────────────────────────────────────────
elif page == "🔮 Predictor":
    st.title("🔮 Predictor Interactivo de Amenazas")
    st.markdown("Ingresa los parámetros de una conexión de red para clasificarla.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🌐 Parámetros de Conexión**")
        duration     = st.number_input("duration",     0, 60000, 0)
        protocol     = st.selectbox("protocol_type", ["tcp", "udp", "icmp"])
        service      = st.selectbox("service", [
            "http", "ftp", "smtp", "ssh", "dns", "ftp_data",
            "other", "private", "domain_u", "telnet"
        ])
        flag         = st.selectbox("flag", ["SF", "S0", "REJ", "RSTO", "SH", "S1", "S2"])
        src_bytes    = st.number_input("src_bytes",    0, 10000000, 0)
        dst_bytes    = st.number_input("dst_bytes",    0, 10000000, 0)
        land         = st.selectbox("land", [0, 1])
        wrong_frag   = st.number_input("wrong_fragment", 0, 10, 0)
        urgent       = st.number_input("urgent",       0, 10, 0)
        hot          = st.number_input("hot",          0, 100, 0)
        logged_in    = st.selectbox("logged_in", [0, 1])
        num_failed   = st.number_input("num_failed_logins", 0, 10, 0)

    with col2:
        st.markdown("**🔧 Parámetros del Host**")
        num_comp     = st.number_input("num_compromised",  0, 1000, 0)
        root_shell   = st.selectbox("root_shell",   [0, 1])
        su_attempted = st.selectbox("su_attempted", [0, 1])
        num_root     = st.number_input("num_root",         0, 1000, 0)
        num_files    = st.number_input("num_file_creations", 0, 100, 0)
        num_shells   = st.number_input("num_shells",       0, 10, 0)
        num_access   = st.number_input("num_access_files", 0, 100, 0)
        num_outbound = st.number_input("num_outbound_cmds", 0, 10, 0)
        is_host_login  = st.selectbox("is_host_login",  [0, 1])
        is_guest_login = st.selectbox("is_guest_login", [0, 1])
        count        = st.number_input("count",    0, 512, 1)
        srv_count    = st.number_input("srv_count", 0, 512, 1)

    with col3:
        st.markdown("**📡 Tasas de Error**")
        serror_rate         = st.slider("serror_rate",         0.0, 1.0, 0.0, 0.01)
        srv_serror_rate     = st.slider("srv_serror_rate",     0.0, 1.0, 0.0, 0.01)
        rerror_rate         = st.slider("rerror_rate",         0.0, 1.0, 0.0, 0.01)
        srv_rerror_rate     = st.slider("srv_rerror_rate",     0.0, 1.0, 0.0, 0.01)
        same_srv_rate       = st.slider("same_srv_rate",       0.0, 1.0, 1.0, 0.01)
        diff_srv_rate       = st.slider("diff_srv_rate",       0.0, 1.0, 0.0, 0.01)
        srv_diff_host_rate  = st.slider("srv_diff_host_rate",  0.0, 1.0, 0.0, 0.01)
        dst_host_count      = st.number_input("dst_host_count",     0, 255, 1)
        dst_host_srv_count  = st.number_input("dst_host_srv_count", 0, 255, 1)
        dst_host_same_srv   = st.slider("dst_host_same_srv_rate",   0.0, 1.0, 1.0, 0.01)
        dst_host_diff_srv   = st.slider("dst_host_diff_srv_rate",   0.0, 1.0, 0.0, 0.01)
        dst_host_src_port   = st.slider("dst_host_same_src_port_rate", 0.0, 1.0, 0.0, 0.01)
        dst_host_srv_diff   = st.slider("dst_host_srv_diff_host_rate", 0.0, 1.0, 0.0, 0.01)
        dst_host_serror     = st.slider("dst_host_serror_rate",     0.0, 1.0, 0.0, 0.01)
        dst_host_srv_serror = st.slider("dst_host_srv_serror_rate", 0.0, 1.0, 0.0, 0.01)
        dst_host_rerror     = st.slider("dst_host_rerror_rate",     0.0, 1.0, 0.0, 0.01)
        dst_host_srv_rerror = st.slider("dst_host_srv_rerror_rate", 0.0, 1.0, 0.0, 0.01)

    st.markdown("---")
    model_pred = st.radio(
        "Modelo para predicción:",
        ["XGBoost", "Random Forest"],
        horizontal=True
    )

    if st.button("🔮 Clasificar Conexión", type="primary", use_container_width=True):
        # Encoding de categóricas
        def safe_encode(encoder, value):
            if value in encoder.classes_:
                return encoder.transform([value])[0]
            return 0

        proto_enc   = safe_encode(encoders["protocol_type"], protocol)
        service_enc = safe_encode(encoders["service"], service)
        flag_enc    = safe_encode(encoders["flag"], flag)

        raw_input = np.array([[
            duration, proto_enc, service_enc, flag_enc,
            src_bytes, dst_bytes, land, wrong_frag, urgent, hot,
            num_failed, logged_in, num_comp, root_shell, su_attempted,
            num_root, num_files, num_shells, num_access, num_outbound,
            is_host_login, is_guest_login, count, srv_count,
            serror_rate, srv_serror_rate, rerror_rate, srv_rerror_rate,
            same_srv_rate, diff_srv_rate, srv_diff_host_rate,
            dst_host_count, dst_host_srv_count, dst_host_same_srv,
            dst_host_diff_srv, dst_host_src_port, dst_host_srv_diff,
            dst_host_serror, dst_host_srv_serror,
            dst_host_rerror, dst_host_srv_rerror
        ]])

        input_scaled = scaler.transform(raw_input)

        model_used = xgb if model_pred == "XGBoost" else rf
        pred_idx   = model_used.predict(input_scaled)[0]
        pred_class = le.classes_[pred_idx]
        proba      = model_used.predict_proba(input_scaled)[0]

        color = CLASS_COLORS.get(pred_class, "#999")
        icon  = CLASS_ICONS.get(pred_class, "❓")
        desc  = CLASS_DESC.get(pred_class, "")

        st.markdown(f"""
        <div style="background:{color}22; border:3px solid {color};
             border-radius:15px; padding:30px; text-align:center; margin:20px 0;">
            <div style="font-size:3rem">{icon}</div>
            <div style="font-size:2rem; font-weight:bold; color:{color}">
                {pred_class.upper()}</div>
            <div style="color:#ccc; margin-top:10px; font-size:1rem">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Probabilidades por clase")
        prob_df = pd.DataFrame({
            "Clase": le.classes_,
            "Probabilidad": proba
        }).sort_values("Probabilidad", ascending=False)

        fig, ax = plt.subplots(figsize=(8, 3))
        colors_bar = [CLASS_COLORS.get(c, "#999") for c in prob_df["Clase"]]
        ax.barh(prob_df["Clase"], prob_df["Probabilidad"], color=colors_bar)
        ax.set_xlim(0, 1)
        ax.set_facecolor("#0d1117"); fig.patch.set_facecolor("#0d1117")
        ax.tick_params(colors="white")
        ax.set_xlabel("Probabilidad", color="white")
        for i, v in enumerate(prob_df["Probabilidad"]):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", color="white", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

# ─────────────────────────────────────────────
# PÁGINA: COMPARATIVA
# ─────────────────────────────────────────────
elif page == "📈 Comparativa":
    st.title("📈 Comparativa de Modelos")
    st.markdown("---")

    y_pred_rf  = rf.predict(X_test)
    y_pred_xgb = xgb.predict(X_test)

    models_data = {
        "Random Forest": {
            "acc": accuracy_score(y_test, y_pred_rf),
            "f1":  f1_score(y_test, y_pred_rf, average="weighted"),
            "f1_macro": f1_score(y_test, y_pred_rf, average="macro"),
        },
        "XGBoost": {
            "acc": accuracy_score(y_test, y_pred_xgb),
            "f1":  f1_score(y_test, y_pred_xgb, average="weighted"),
            "f1_macro": f1_score(y_test, y_pred_xgb, average="macro"),
        },
    }

    col1, col2 = st.columns(2)
    for i, (name, data) in enumerate(models_data.items()):
        with (col1 if i == 0 else col2):
            st.markdown(f"#### {name}")
            st.metric("Accuracy",       f"{data['acc']:.4f}")
            st.metric("F1 Weighted",    f"{data['f1']:.4f}")
            st.metric("F1 Macro",       f"{data['f1_macro']:.4f}")

    st.markdown("---")
    st.markdown("#### Gráfica Comparativa")
    img_path = os.path.join(REPORTS_DIR, "03_model_comparison.png")
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)

    st.markdown("---")
    st.markdown("#### F1-Score por Clase")
    classes = list(le.classes_)
    f1_rf_per  = f1_score(y_test, y_pred_rf,  average=None)
    f1_xgb_per = f1_score(y_test, y_pred_xgb, average=None)

    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, f1_rf_per,  0.35, label="Random Forest", color="#2196F3")
    ax.bar(x + 0.2, f1_xgb_per, 0.35, label="XGBoost",       color="#4CAF50")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, color="white")
    ax.set_ylim(0, 1.1)
    ax.set_title("F1-Score por Clase", color="white")
    ax.legend()
    ax.set_facecolor("#0d1117"); fig.patch.set_facecolor("#0d1117")
    ax.tick_params(colors="white")
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", color="white", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig); plt.close()