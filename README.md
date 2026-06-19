# AutoML-Lite 🤖

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.30%2B-FF4B4B)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A Streamlit app that turns any tabular dataset (CSV or Excel) into a trained, compared,
and exportable machine learning model — no code required from the end user.

> Upload data → pick a target column → train one model or a whole leaderboard of them →
> inspect diagnostics and feature importance → download the pipeline as a ready-to-use `.pkl`.

## ✨ Features

- 📂 **Upload CSV or Excel** (`.csv`, `.xlsx`, `.xls`, `.xlsm`) — multi-sheet workbooks let
  you pick a sheet
- 📊 **Automatic data profiling**: missing values, dtypes, duplicate rows, correlation heatmap
- 🧠 **Auto-detects classification vs. regression** from the target column (override if needed)
- 🗂️ **Three training modes**:
  | Tab | What it does |
  |---|---|
  | 🏆 Auto Leaderboard | Trains every available model for the detected task and ranks them |
  | 🧮 Classification | Pick one specific classifier (Logistic Regression, Random Forest, SVM, Decision Tree, kNN, XGBoost), a target column, and a test size |
  | 📈 Regression | Pick one specific regressor (Linear Regression, Random Forest, SVM, Decision Tree, XGBoost), a target column, and a test size |
- 📉 **Diagnostics**: confusion matrix (classification) or residual / actual-vs-predicted
  plots (regression)
- ⭐ **Feature importance** chart for the trained model
- 💾 **Download the trained pipeline** as a `.pkl` — preprocessing and model bundled together,
  ready for reuse in any Python script

## 📁 Project Structure

```
automl_lite/
├── app.py                 # Streamlit GUI
├── pipeline/
│   ├── __init__.py
│   └── core.py             # Profiling, preprocessing, training, evaluation logic
├── requirements.txt
└── README.md
```

The split between `app.py` (UI) and `pipeline/core.py` (logic) is intentional — it keeps
the ML pipeline testable and reusable outside of Streamlit (notebooks, scripts, APIs).

## 🚀 Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/automl-lite.git
cd automl-lite

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## 🖥️ Usage

1. Upload a CSV/Excel file (or pick a demo dataset) in the sidebar.
2. Review the auto-generated data profile and correlation heatmap.
3. Choose a tab:
   - **Auto Leaderboard** to compare every model at once, or
   - **Classification** / **Regression** to train one specific model.
4. Pick the target column, test size, and (in the single-model tabs) the model.
5. Click **Train**. Explore the leaderboard, confusion matrix / residual plots, and
   feature importance.
6. Download the winning pipeline as a `.pkl`:

```python
import joblib
import pandas as pd

pipeline = joblib.load("random_forest_pipeline.pkl")
new_data = pd.read_csv("new_rows.csv")
predictions = pipeline.predict(new_data)
```

## 🛣️ Roadmap / possible extensions

- [ ] Hyperparameter tuning (`GridSearchCV` / `Optuna`) with a live progress bar
- [ ] SHAP-based explainability instead of raw feature importances
- [ ] Cross-validation instead of a single train/test split
- [ ] Unsupervised exploration tab (k-means clustering, PCA)
- [ ] Time-series-aware splitting
- [ ] Dockerfile + deployment to Streamlit Community Cloud

## 🧰 Tech Stack

Streamlit · scikit-learn · pandas · plotly · XGBoost · joblib · openpyxl

## 📄 License

MIT — feel free to fork, modify, and use for your own portfolio or projects.
