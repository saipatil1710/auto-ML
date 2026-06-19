"""
Core AutoML pipeline logic: data profiling, preprocessing, model training, and evaluation.
Kept separate from the Streamlit app so it can be tested/reused independently.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data profiling
# ---------------------------------------------------------------------------

def profile_dataframe(df: pd.DataFrame) -> dict:
    """Return a summary profile of the dataframe: dtypes, missing values, stats."""
    profile = {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "missing": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_cols": df.select_dtypes(include=np.number).columns.tolist(),
        "categorical_cols": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "n_duplicates": int(df.duplicated().sum()),
    }
    return profile


def infer_task_type(y: pd.Series) -> str:
    """Infer whether the target column implies classification or regression."""
    if y.dtype == "object" or y.dtype.name == "category":
        return "classification"
    n_unique = y.nunique()
    # Heuristic: few unique values relative to dataset size -> classification
    if n_unique <= max(20, int(0.05 * len(y))) and n_unique < len(y) * 0.5:
        return "classification"
    return "regression"


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer that imputes + scales numeric, imputes + one-hot encodes categorical."""
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])

    return preprocessor


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------

def get_models(task_type: str) -> dict:
    """Return a dict of {model_name: estimator} appropriate for the task."""
    if task_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "kNN": KNeighborsClassifier(),
        }
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBClassifier(eval_metric="logloss", random_state=42)
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
            "SVM": SVR(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
        }
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBRegressor(random_state=42)
    return models


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(df: pd.DataFrame, target_col: str, task_type: str,
                        test_size: float = 0.2, random_state: int = 42,
                        model_names: list = None) -> dict:
    """
    Train candidate models on the dataset and return results, fitted pipelines,
    and test data for further inspection (e.g. confusion matrix, feature importance).

    If model_names is provided, only those models are trained (must match keys
    returned by get_models()). Otherwise all available models for the task are trained.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if task_type == "classification" and y.dtype == "object":
        y = y.astype("category")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if task_type == "classification" else None
    )

    preprocessor = build_preprocessor(X)
    all_models = get_models(task_type)

    if model_names:
        models = {name: all_models[name] for name in model_names if name in all_models}
        if not models:
            raise ValueError(f"None of the requested models {model_names} are available for task '{task_type}'.")
    else:
        models = all_models

    results = []
    fitted_pipelines = {}

    # XGBoost's classifier requires numeric (0..n-1) labels, not raw strings.
    le = LabelEncoder()
    if task_type == "classification":
        le.fit(y)

    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        needs_encoding = task_type == "classification" and name == "XGBoost"
        if needs_encoding:
            pipe.fit(X_train, le.transform(y_train))
            y_pred = le.inverse_transform(pipe.predict(X_test))
        else:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

        if task_type == "classification":
            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1": f1_score(y_test, y_pred, average="weighted"),
                "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, y_pred, average="weighted"),
            }
        else:
            metrics = {
                "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
                "MAE": mean_absolute_error(y_test, y_pred),
                "R2": r2_score(y_test, y_pred),
            }

        results.append({"Model": name, **metrics})
        fitted_pipelines[name] = pipe

    results_df = pd.DataFrame(results)
    sort_col = "F1" if task_type == "classification" else "R2"
    results_df = results_df.sort_values(by=sort_col, ascending=False).reset_index(drop=True)

    return {
        "results_df": results_df,
        "fitted_pipelines": fitted_pipelines,
        "X_test": X_test,
        "y_test": y_test,
        "task_type": task_type,
    }


def get_feature_importance(pipeline: Pipeline, top_n: int = 15) -> pd.DataFrame:
    """Extract feature importances (tree models) or coefficients (linear models) if available."""
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(model.n_features_in_)]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
    else:
        return pd.DataFrame()

    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False).head(top_n)
    return fi_df
