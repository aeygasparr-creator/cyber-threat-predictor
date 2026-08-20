[idfdvgk4v3ookngv0emzs-6532622b_ZZ93Rcul7D.md](https://github.com/user-attachments/files/31244885/idfdvgk4v3ookngv0emzs-6532622b_ZZ93Rcul7D.md)# 🛡️ Cyber Threat Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11.9-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=for-the-badge)

**Sistema de Detección de Intrusiones (IDS) basado en Machine Learning**  
*Clasificación de amenazas de red en tiempo real sobre el dataset NSL-KDD*

[![🚀 Ver Demo en Vivo](https://img.shields.io/badge/🚀_Demo_en_Vivo-Streamlit_Cloud-FF4B4B?style=for-the-badge)](https://cyber-threat-predictor-ttmhyby9qvtnc64h68onds.streamlit.app)

</div>

---

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Demo en Vivo](#-demo-en-vivo)
- [Resultados](#-resultados-de-modelos)
- [Categorías de Amenaza](#-categorías-de-amenaza)
- [Stack Tecnológico](#-stack-tecnológico)
- [Arquitectura](#-arquitectura-del-pipeline)
- [Instalación](#-instalación-local)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Dataset](#-dataset-nsl-kdd)
- [Autor](#-autor)

---

## 📖 Descripción

**Cyber Threat Predictor** es un sistema de detección de intrusiones (IDS) que clasifica el tráfico de red en 5 categorías de amenaza usando modelos de Machine Learning entrenados sobre el dataset **NSL-KDD** (148,517 registros).

El proyecto incluye:
- 🤖 **3 modelos** entrenados: XGBoost, Random Forest y Regresión Logística
- ⚖️ **Balanceo de clases** con técnica SMOTE
- 📊 **Dashboard interactivo** construido con Streamlit
- 🔮 **Predictor en tiempo real** con parámetros de red personalizables
- 📈 **Análisis exploratorio** de datos completo

---

## 🚀 Demo en Vivo

> 🌐 **[https://cyber-threat-predictor-ttmhyby9qvtnc64h68onds.streamlit.app](https://cyber-threat-predictor-ttmhyby9qvtnc64h68onds.streamlit.app)**

El dashboard incluye 5 secciones:

| Sección | Descripción |
|---------|-------------|
| 🏠 **Inicio** | Resumen general del sistema y métricas clave |
| 📊 **EDA** | Análisis exploratorio con distribuciones Train/Test |
| 🎯 **Modelos** | Evaluación con matriz de confusión y reporte de clasificación |
| 🔮 **Predictor** | Clasificación interactiva de conexiones de red |
| 📉 **Comparativa** | Comparación visual entre modelos |

---

## 🏆 Resultados de Modelos

<div align="center">

| Modelo | Accuracy | F1-Score (weighted) | F1-Score (macro) |
|--------|----------|---------------------|------------------|
| 🥇 **XGBoost** | **0.7888** | **0.7575** | **0.6363** |
| 🥈 **Random Forest** | 0.7391 | 0.6929 | 0.5040 |
| 🥉 **Regresión Logística** | — | — | — |

</div>

> ⚖️ Se aplicó **SMOTE** (Synthetic Minority Over-sampling Technique) para balancear las clases minoritarias R2L y U2R.

---

## 🛡️ Categorías de Amenaza

| Categoría | Descripción |
|-----------|-------------|
| ✅ **Normal** | Tráfico legítimo de red |
| 💥 **DoS** | Denegación de Servicio — satura recursos del sistema |
| 🔍 **Probe** | Escaneo y reconocimiento de vulnerabilidades |
| 🔓 **R2L** | Remote-to-Local — acceso remoto no autorizado |
| ⚠️ **U2R** | User-to-Root — escalada de privilegios |

---

## ⚙️ Stack Tecnológico

```
Python 3.11.9      → Lenguaje principal
Scikit-learn       → Pipeline de ML y métricas
XGBoost            → Modelo principal (mejor accuracy)
Pandas / NumPy     → Procesamiento de datos
Imbalanced-learn   → Técnica SMOTE
Streamlit          → Dashboard interactivo
Joblib             → Serialización de modelos
Matplotlib         → Visualizaciones
```

---

## 🔄 Arquitectura del Pipeline

```
NSL-KDD (raw)
    │
    ▼
preprocess.py ──► Encoding + Scaling + SMOTE
    │
    ▼
train.py ──► XGBoost | Random Forest | Logistic Regression
    │
    ▼
models/ ──► scaler.pkl | encoders.pkl | model_xgb.pkl | model_rf.pkl
    │
    ▼
app.py ──► Dashboard Streamlit (Deploy en Streamlit Cloud)
```

---

## 💻 Instalación Local

```bash
# 1. Clonar el repositorio
git clone https://github.com/aeygasparr-creator/cyber-threat-predictor.git
cd cyber-threat-predictor

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación
streamlit run src/app.py
```

---

## 📁 Estructura del Proyecto

```
cyber-threat-predictor/
│
├── src/
│   ├── app.py              # Dashboard principal Streamlit
│   ├── preprocess.py       # Preprocesamiento de datos
│   └── train.py            # Entrenamiento de modelos
│
├── reports/
│   └── figures/            # Gráficas y visualizaciones
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📦 Dataset NSL-KDD

| Característica | Valor |
|----------------|-------|
| **Total registros** | 148,517 |
| **Features** | 41 |
| **Clases** | 5 (Normal, DoS, Probe, R2L, U2R) |
| **Train set** | 125,973 registros |
| **Test set** | 22,544 registros |
| **Fuente** | [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html) |

---

## 👨‍💻 Autor

<div align="center">

**Alejandro Eduardo Gaspar Rivera**  
*Full Stack Developer | Data Science & Machine Learning*  
*Ingeniería de Sistemas — UNAC*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-gaspar--rivera--alejandro-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/gaspar-rivera-alejandro)
[![GitHub](https://img.shields.io/badge/GitHub-aeygasparr--creator-181717?style=for-the-badge&logo=github)](https://github.com/aeygasparr-creator)

</div>

---

<div align="center">

⭐ **Si este proyecto te fue útil, dale una estrella!** ⭐

*Made with ❤️ and Python in Lima, Perú 🇵🇪*

</div>
