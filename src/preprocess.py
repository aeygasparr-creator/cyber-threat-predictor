"""
preprocess.py
=============
Carga, limpia y prepara los datasets NSL-KDD para entrenamiento.
Fuentes: KDDTrain+.txt y KDDTest+.txt (etiquetas completas)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# ─────────────────────────────────────────────
# COLUMNAS DEL DATASET NSL-KDD
# ─────────────────────────────────────────────
COLUMNS = [
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
    "label", "difficulty"
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

RAW_DIR       = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR    = os.path.join("models")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,    exist_ok=True)

# ─────────────────────────────────────────────
# MAPEO DE CATEGORÍAS DE ATAQUE
# ─────────────────────────────────────────────
ATTACK_CATEGORIES = {
    "normal": "normal",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS",
    "apache2": "DoS", "processtable": "DoS", "udpstorm": "DoS",
    # Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    # R2L
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L",
    "warezclient": "R2L", "warezmaster": "R2L", "sendmail": "R2L",
    "named": "R2L", "snmpgetattack": "R2L", "snmpguess": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "worm": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "httptunnel": "U2R", "ps": "U2R",
    "sqlattack": "U2R", "xterm": "U2R",
}


# ─────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────
def load_txt(path, name="dataset"):
    """Carga un archivo .txt del NSL-KDD (train o test)."""
    print(f"[INFO] Cargando {name} desde: {path}")
    df = pd.read_csv(path, header=None, names=COLUMNS)

    # Limpiar etiquetas: espacios, comillas y puntos finales
    df["label"] = (df["label"].astype(str)
                              .str.strip()
                              .str.strip("'\"")
                              .str.rstrip("."))

    print(f"[INFO] {name} shape: {df.shape}")
    print(f"[DEBUG] Etiquetas únicas ({len(df['label'].unique())}): "
          f"{sorted(df['label'].unique())}\n")
    return df


# ─────────────────────────────────────────────
# 2. ETIQUETADO BINARIO Y MULTICLASE
# ─────────────────────────────────────────────
def add_labels(df, name="dataset"):
    df = df.copy()

    df["label_binary"] = df["label"].apply(
        lambda x: 0 if x == "normal" else 1
    )
    df["label_multi"] = df["label"].apply(
        lambda x: ATTACK_CATEGORIES.get(x, "Other")
    )

    print(f"[INFO] {name} — distribución multiclase:")
    print(df["label_multi"].value_counts().to_string())
    print(f"\n[INFO] {name} — binario: "
          f"{df['label_binary'].value_counts().sort_index().to_dict()} "
          f"(0=normal, 1=ataque)\n")
    return df


# ─────────────────────────────────────────────
# 3. ENCODING Y ESCALADO
# ─────────────────────────────────────────────
def encode_and_scale(train_df, test_df):
    encoders = {}

    # Encoding de columnas categóricas
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        # Valores desconocidos en test → primera clase conocida
        test_df[col] = test_df[col].astype(str).apply(
            lambda x: x if x in le.classes_ else le.classes_[0]
        )
        test_df[col] = le.transform(test_df[col])
        encoders[col] = le
        print(f"[INFO] Encoded '{col}': {len(le.classes_)} clases")

    joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.pkl"))
    print("[INFO] Encoders guardados en models/encoders.pkl")

    # Columnas de features (sin label ni difficulty)
    feature_cols = [c for c in COLUMNS if c not in
                    ["label", "difficulty"]]

    # Escalado
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols]  = scaler.transform(test_df[feature_cols])

    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    print("[INFO] Scaler guardado en models/scaler.pkl\n")

    return train_df, test_df, feature_cols


# ─────────────────────────────────────────────
# 4. PIPELINE COMPLETO
# ─────────────────────────────────────────────
def run_pipeline():
    print("\n" + "="*50)
    print("  PIPELINE DE PREPROCESAMIENTO NSL-KDD")
    print("="*50 + "\n")

    # ── Cargar
    train = load_txt(os.path.join(RAW_DIR, "KDDTrain+.txt"), name="TRAIN")
    test  = load_txt(os.path.join(RAW_DIR, "KDDTest+.txt"),  name="TEST")

    # ── Etiquetar
    train = add_labels(train, name="TRAIN")
    test  = add_labels(test,  name="TEST")

    # ── Encode + Scale
    train, test, feature_cols = encode_and_scale(train, test)

    # ── Guardar CSVs procesados
    train_out = os.path.join(PROCESSED_DIR, "train_processed.csv")
    test_out  = os.path.join(PROCESSED_DIR, "test_processed.csv")
    train.to_csv(train_out, index=False)
    test.to_csv(test_out,   index=False)

    print(f"[✅] Train procesado guardado: {train_out}")
    print(f"[✅] Test procesado guardado:  {test_out}")
    print(f"\n[INFO] Features usadas: {len(feature_cols)} columnas")
    print(f"[INFO] Train: {train.shape[0]:,} registros")
    print(f"[INFO] Test:  {test.shape[0]:,} registros")
    print("\n" + "="*50)
    print("  PREPROCESAMIENTO COMPLETADO ✅")
    print("="*50 + "\n")

    return train, test, feature_cols


if __name__ == "__main__":
    run_pipeline()