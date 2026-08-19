"""
train.py
========
EDA + Entrenamiento con SMOTE para clases desbalanceadas.
Modelos: Random Forest, XGBoost, Logistic Regression
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────
PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR    = os.path.join("models")
REPORTS_DIR   = os.path.join("reports", "figures")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

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

KNOWN_CLASSES = ["DoS", "Probe", "R2L", "U2R", "normal"]


# ─────────────────────────────────────────────
# 1. CARGA
# ─────────────────────────────────────────────
def load_data():
    print("[INFO] Cargando datos procesados...")
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"))
    test  = pd.read_csv(os.path.join(PROCESSED_DIR, "test_processed.csv"))
    print(f"[INFO] Train: {train.shape} | Test: {test.shape}")
    return train, test


# ─────────────────────────────────────────────
# 2. EDA
# ─────────────────────────────────────────────
def run_eda(train):
    print("\n[INFO] Generando gráficas EDA...")
    sns.set_theme(style="darkgrid", palette="muted")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    counts = train["label_multi"].value_counts()
    colors = ["#4CAF50", "#F44336", "#2196F3", "#FF9800", "#9C27B0"]
    axes[0].bar(counts.index, counts.values, color=colors[:len(counts)])
    axes[0].set_title("Distribución de Categorías de Ataque", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Categoría")
    axes[0].set_ylabel("Cantidad")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 500, f"{v:,}", ha="center", fontsize=9)

    bin_counts = train["label_binary"].value_counts().sort_index()
    axes[1].pie(bin_counts.values,
                labels=["Normal", "Ataque"],
                autopct="%1.1f%%", colors=["#4CAF50", "#F44336"],
                startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[1].set_title("Normal vs Ataque (Binario)", fontsize=13, fontweight="bold")

    plt.tight_layout()
    path1 = os.path.join(REPORTS_DIR, "01_class_distribution.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✅] Guardado: {path1}")

    variances = train[FEATURE_COLS].var().sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(variances.index[::-1], variances.values[::-1], color="#2196F3")
    ax.set_title("Top 15 Features por Varianza", fontsize=13, fontweight="bold")
    ax.set_xlabel("Varianza")
    plt.tight_layout()
    path2 = os.path.join(REPORTS_DIR, "02_feature_variance.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✅] Guardado: {path2}")
    print("[INFO] EDA completado.\n")


# ─────────────────────────────────────────────
# 3. ENTRENAMIENTO
# ─────────────────────────────────────────────
def train_models(train, test):
    X_train = train[FEATURE_COLS].values
    X_test  = test[FEATURE_COLS].values

    # LabelEncoder fit con clases conocidas
    le = LabelEncoder()
    le.fit(KNOWN_CLASSES)

    y_train_raw = train["label_multi"].apply(
        lambda x: x if x in le.classes_ else "normal"
    )
    y_train_multi = le.transform(y_train_raw)

    y_test_raw = test["label_multi"].apply(
        lambda x: x if x in le.classes_ else "normal"
    )
    y_test_multi = le.transform(y_test_raw)

    y_train_bin = train["label_binary"].values
    y_test_bin  = test["label_binary"].values

    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder_multi.pkl"))

    print(f"[INFO] Clases: {list(le.classes_)}")
    print(f"[INFO] Distribución train ANTES de SMOTE:")
    unique, counts = np.unique(y_train_multi, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"       {le.classes_[u]:10s}: {c:,}")

    # ── SMOTE: balancear clases minoritarias
    print("\n[INFO] Aplicando SMOTE...")
    smote = SMOTE(
        sampling_strategy={
            2: 5000,   # R2L  995  → 5,000
            3: 2000,   # U2R   52  → 2,000
        },
        random_state=42,
        k_neighbors=5
    )
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train_multi)

    print(f"[INFO] Distribución train DESPUÉS de SMOTE:")
    unique2, counts2 = np.unique(y_train_sm, return_counts=True)
    for u, c in zip(unique2, counts2):
        print(f"       {le.classes_[u]:10s}: {c:,}")
    print()

    results = {}

    # ── Modelo 1: Random Forest
    print("="*50)
    print("[1/3] Entrenando Random Forest (multiclase + SMOTE)...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    rf.fit(X_train_sm, y_train_sm)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test_multi, y_pred_rf)
    f1_rf  = f1_score(y_test_multi, y_pred_rf, average="weighted")
    print(f"[✅] Random Forest — Accuracy: {acc_rf:.4f} | F1: {f1_rf:.4f}")
    print(classification_report(y_test_multi, y_pred_rf, target_names=le.classes_))
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.pkl"))
    results["Random Forest"] = {"acc": acc_rf, "f1": f1_rf,
                                 "y_pred": y_pred_rf, "y_true": y_test_multi}

    # ── Modelo 2: XGBoost
    print("="*50)
    print("[2/3] Entrenando XGBoost (multiclase + SMOTE)...")

    # Pesos por clase para XGBoost
    class_counts = dict(zip(unique2, counts2))
    total = sum(class_counts.values())
    sample_weights = np.array([
        total / (len(class_counts) * class_counts[y]) for y in y_train_sm
    ])

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss",
        n_jobs=-1,
        verbosity=0
    )
    xgb.fit(X_train_sm, y_train_sm, sample_weight=sample_weights)
    y_pred_xgb = xgb.predict(X_test)
    acc_xgb = accuracy_score(y_test_multi, y_pred_xgb)
    f1_xgb  = f1_score(y_test_multi, y_pred_xgb, average="weighted")
    print(f"[✅] XGBoost — Accuracy: {acc_xgb:.4f} | F1: {f1_xgb:.4f}")
    print(classification_report(y_test_multi, y_pred_xgb, target_names=le.classes_))
    joblib.dump(xgb, os.path.join(MODELS_DIR, "xgboost.pkl"))
    results["XGBoost"] = {"acc": acc_xgb, "f1": f1_xgb,
                           "y_pred": y_pred_xgb, "y_true": y_test_multi}

    # ── Modelo 3: Logistic Regression (binario)
    print("="*50)
    print("[3/3] Entrenando Logistic Regression (binario)...")
    lr = LogisticRegression(
        max_iter=1000, random_state=42,
        class_weight="balanced", n_jobs=-1
    )
    lr.fit(X_train, y_train_bin)
    y_pred_lr = lr.predict(X_test)
    acc_lr = accuracy_score(y_test_bin, y_pred_lr)
    f1_lr  = f1_score(y_test_bin, y_pred_lr, average="weighted")
    print(f"[✅] Logistic Regression — Accuracy: {acc_lr:.4f} | F1: {f1_lr:.4f}")
    print(classification_report(y_test_bin, y_pred_lr,
                                 target_names=["Normal", "Ataque"]))
    joblib.dump(lr, os.path.join(MODELS_DIR, "logistic_regression.pkl"))

    return results, le, X_test, y_test_multi


# ─────────────────────────────────────────────
# 4. GRÁFICAS DE RESULTADOS
# ─────────────────────────────────────────────
def plot_results(results, le, X_test, y_test_multi):
    print("\n[INFO] Generando gráficas de resultados...")

    # Comparación de modelos
    names = list(results.keys())
    accs  = [results[m]["acc"] for m in names]
    f1s   = [results[m]["f1"]  for m in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - 0.2, accs, 0.35, label="Accuracy", color="#2196F3")
    bars2 = ax.bar(x + 0.2, f1s,  0.35, label="F1-Score",  color="#4CAF50")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title("Comparación de Modelos", fontsize=13, fontweight="bold")
    ax.legend()
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    path3 = os.path.join(REPORTS_DIR, "03_model_comparison.png")
    plt.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✅] Guardado: {path3}")

    # Matriz de confusión del mejor modelo
    best = max(results, key=lambda m: results[m]["f1"])
    cm = confusion_matrix(results[best]["y_true"], results[best]["y_pred"])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title(f"Matriz de Confusión — {best}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Real")
    ax.set_xlabel("Predicho")
    plt.tight_layout()
    path4 = os.path.join(REPORTS_DIR, "04_confusion_matrix.png")
    plt.savefig(path4, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✅] Guardado: {path4}")

    # Feature importance — Random Forest
    rf = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS)
    top15 = importances.sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(top15.index[::-1], top15.values[::-1], color="#9C27B0")
    ax.set_title("Top 15 Features — Random Forest", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importancia")
    plt.tight_layout()
    path5 = os.path.join(REPORTS_DIR, "05_feature_importance.png")
    plt.savefig(path5, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✅] Guardado: {path5}")


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  ENTRENAMIENTO DE MODELOS — CYBER THREAT PREDICTOR")
    print("="*50)

    train, test = load_data()
    run_eda(train)
    results, le, X_test, y_test_multi = train_models(train, test)
    plot_results(results, le, X_test, y_test_multi)

    print("\n" + "="*50)
    print("  ENTRENAMIENTO COMPLETADO ✅")
    print("="*50)
    print("\n📁 Modelos guardados en:  models/")
    print("📊 Gráficas guardadas en: reports/figures/")