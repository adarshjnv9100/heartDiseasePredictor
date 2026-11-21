# ❤️ Heart Disease Prediction Web App

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67.0-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning web application designed to predict the likelihood of cardiovascular disease based on clinical and biometric parameters. Built with **scikit-learn** and deployed as an interactive dashboard using **Streamlit**.

---

## 📌 Project Overview

Cardiovascular diseases (CVDs) are the leading cause of death globally. Early detection and risk assessment are critical for preventative care. This project implements a complete data science lifecycle:
1. **Exploratory Data Analysis (EDA)** and visualization of patient health factors.
2. **Data Preprocessing**: Handling categorical encodings (`drop_first=True`) and standardizing feature distributions (`StandardScaler`).
3. **Model Selection & Benchmarking**: Evaluating multiple classifiers (Logistic Regression, KNN, Naive Bayes, Decision Trees, SVM).
4. **Interactive Deployment**: A responsive, real-time diagnostic dashboard built with Streamlit.

---

## 📊 Dataset & Features

The model is trained on a comprehensive heart disease dataset comprising **918 patient observations** across 11 clinical features:

| Feature | Description | Values / Range |
| :--- | :--- | :--- |
| **Age** | Age of the patient | 18 – 100 years |
| **Sex** | Biological sex | Male (`M`), Female (`F`) |
| **ChestPainType** | Type of chest pain | `ATA` (Atypical Angina), `NAP` (Non-Anginal), `ASY` (Asymptomatic), `TA` (Typical Angina) |
| **RestingBP** | Resting blood pressure | 80 – 200 mm Hg |
| **Cholesterol** | Serum cholesterol level | 100 – 600 mg/dL |
| **FastingBS** | Fasting blood sugar > 120 mg/dL | `1` = Yes, `0` = No |
| **RestingECG** | Resting electrocardiogram results | `Normal`, `ST` (ST-T wave abnormality), `LVH` (Left ventricular hypertrophy) |
| **MaxHR** | Maximum heart rate achieved during exercise | 60 – 220 bpm |
| **ExerciseAngina**| Exercise-induced angina | `Y` (Yes), `N` (No) |
| **Oldpeak** | ST depression induced by exercise relative to rest | 0.0 – 6.0 |
| **ST_Slope** | Peak exercise ST segment slope | `Up` (Upsloping), `Flat` (Flat), `Down` (Downsloping) |
| **HeartDisease** *(Target)* | Diagnosis of heart disease | `1` = High Risk, `0` = Low Risk |

---

## 🏆 Model Benchmarking & Performance

Multiple classification algorithms were benchmarked on normalized test data. The **K-Nearest Neighbors (KNN)** model demonstrated the highest predictive accuracy and F1 score:

| Model | Accuracy | F1 Score |
| :--- | :---: | :---: |
| 🥇 **K-Nearest Neighbors (KNN)** | **88.59%** | **0.8986** |
| 🥈 Logistic Regression | 87.50% | 0.8878 |
| 🥉 Naive Bayes | 86.96% | 0.8788 |
| SVM (RBF Kernel) | 86.41% | 0.8804 |
| Decision Tree | 74.46% | 0.7513 |

---

## 📁 Repository Structure

```text
heartDiseasePredictor/
├── .gitignore               # Excludes virtual environments, pycache, and checkpoints
├── app.py                   # Streamlit web application
├── HeartdiseaseFinal.ipynb  # Comprehensive EDA, training & evaluation notebook
├── heart.csv                # Dataset source
├── KNN_heart.pkl            # Serialized trained KNN model
├── scaler.pkl               # Serialized StandardScaler
├── columns.pkl              # Feature order & dummy column references
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

Follow these instructions to run the application locally on your machine.

### Prerequisites
- Python 3.10+
- Git

### 1. Clone the repository
```bash
git clone https://github.com/adarshjnv9100/heartDiseasePredictor.git
cd heartDiseasePredictor
```

### 2. Set up a virtual environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install required packages
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit application
```bash
streamlit run app.py
```
After running this command, open your web browser and navigate to:
```text
http://localhost:8501
```

---

## 💡 How It Works

1. **User Input:** Enter patient medical parameters using the intuitive sliders and dropdown selectors.
2. **One-Hot Encoding:** Input values are dynamically aligned with the 15 encoded features used during training.
3. **Scaling:** Values are normalized using the saved `StandardScaler` artifact.
4. **Prediction:** The KNN model evaluates the sample against nearest neighbors and outputs:
   - Prediction label: **High Risk** or **Low Risk**
   - Probability score percentage
   - Health recommendation guidance

---

## ⚠️ Medical Disclaimer

This application is created for **educational and portfolio demonstration purposes only**. It should not be used as a substitute for professional medical diagnosis, advice, or treatment.

---

## 👤 Author

Developed by **[Adarsh](https://github.com/adarshjnv9100)**  
If you find this project helpful, please consider giving it a ⭐ on GitHub!
