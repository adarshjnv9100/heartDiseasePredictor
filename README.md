# Heart Disease Prediction App

A machine learning web application built with [Streamlit](https://streamlit.io/) and [scikit-learn](https://scikit-learn.org/) to predict the likelihood of heart disease based on patient clinical indicators.

## Features
- Interactive Streamlit user interface with sliders and dropdown menus
- K-Nearest Neighbors (KNN) classification model
- Feature scaling with StandardScaler
- Immediate risk assessment: **High Risk** or **Low Risk**

## Project Structure
```text
├── app.py                   # Streamlit web application
├── HeartdiseaseFinal.ipynb  # EDA, model training, and evaluation notebook
├── heart.csv                # Dataset
├── KNN_heart.pkl            # Trained KNN model
├── scaler.pkl               # Fitted StandardScaler
├── columns.pkl              # Feature columns list
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Getting Started

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd heartDisease
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.
