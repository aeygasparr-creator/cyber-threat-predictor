[README.md - Cyber Threat Predictor.md](https://github.com/user-attachments/files/31207102/README.md.-.Cyber.Threat.Predictor.md)
# 🛡️ Cyber Threat Predictor

> Sistema de clasificación de amenazas cibernéticas con Machine Learning, entrenado sobre el dataset NSL-KDD.

![Python](https://img.shields.io/badge/Python-3.11.9-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Descripción

Cyber Threat Predictor es un sistema de detección de intrusiones (IDS) que clasifica el tráfico de red en categorías de amenaza usando modelos de Machine Learning. Incluye un dashboard interactivo construido con Streamlit para visualización en tiempo real.

---

## 🚀 Demo

> Dashboard corriendo en `localhost:8501` con clasificación en tiempo real.

---

## 🧠 Modelos Entrenados

| Modelo | Exactitud |
|--------|-----------|
| XGBoost | **0.7888** |
| Random Forest | 0.7391 |

---

## 📊 Dataset: NSL-KDD

| Split | Registros |
|-------|-----------|
| Train | 125,973 |
| Test | 22,544 |

Categorías de amenaza clasificadas:
- `Normal` — Tráfico legítimo
- `DoS` — Denial of Service
- `Probe` — Escaneo/Reconocimiento
- `R2L` — Remote to Local
- `U2R` — User to Root

---

## 🛠️ Stack Tecnológico

- **Lenguaje:** Python 3.11.9
- **ML:** Scikit-learn, XGBoost, Imbalanced-learn (SMOTE)
- **Data:** Pandas, NumPy
- **Dashboard:** Streamlit
- **Serialización:** Joblib

---

## 📁 Estructura del Proyecto

```
cyber-threat-predictor/
├── src/
│   ├── preprocess.py      # Preprocesamiento del dataset NSL-KDD
│   ├── train.py           # Entrenamiento de modelos con SMOTE
│   └── app.py             # Dashboard Streamlit
├── reports/
│   └── figures/           # Gráficas de resultados
├── requirements.txt
└── .gitignore
```

> ⚠️ La carpeta `data/` no está incluida en el repositorio por el tamaño de los archivos.

---

## ⚙️ Instalación y Uso

### 1. Clonar el repositorio
```bash
git clone https://github.com/aeygasparr-creator/cyber-threat-predictor.git
cd cyber-threat-predictor
```

### 2. Crear entorno virtual e instalar dependencias
```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 3. Descargar el dataset NSL-KDD
Descarga los archivos desde [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/nsl.html) y colócalos en:
```
data/raw/KDDTrain+.txt
data/raw/KDDTest+.txt
```

### 4. Preprocesar y entrenar
```bash
python src/preprocess.py
python src/train.py
```

### 5. Lanzar el dashboard
```bash
streamlit run src/app.py
```

---

## 👤 Autor

**Alejandro Eduardo Gaspar Rivera**
- 💼 [LinkedIn](https://linkedin.com/in/gaspar-rivera-alejandro)
- 🐙 [GitHub](https://github.com/aeygasparr-creator)

---

## 📄 Licencia

MIT License — libre para usar y modificar.
