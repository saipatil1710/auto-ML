"""
AutoML-Lite: Streamlit GUI for uploading tabular data, training multiple models,
and comparing results -- no code required from the end user.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import joblib
import io

from pipeline.core import (
    profile_dataframe, infer_task_type, train_and_evaluate,
    get_feature_importance, get_models
)
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="AutoML-Lite", layout="wide", page_icon="🤖")

st.title("🤖 AutoML-Lite")
st.caption("Upload any tabular dataset, pick a target column, and compare models automatically.")

# ---------------------------------------------------------------------------
# Sidebar: data source
# ---------------------------------------------------------------------------

st.sidebar.header("1. Load Data")
data_source = st.sidebar.radio("Choose a data source:", ["Upload File", "Use demo dataset"])

df = None

if data_source == "Upload File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload a CSV or Excel file",
        type=["csv", "xlsx", "xls", "xlsm"],
    )
    if uploaded_file is not None:
        filename = uploaded_file.name.lower()
        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                # Excel files may have multiple sheets -- let the user pick one.
                excel_file = pd.ExcelFile(uploaded_file)
                sheet_names = excel_file.sheet_names
                if len(sheet_names) > 1:
                    sheet = st.sidebar.selectbox("Select sheet:", sheet_names)
                else:
                    sheet = sheet_names[0]
                df = pd.read_excel(excel_file, sheet_name=sheet)
        except Exception as e:
            st.sidebar.error(f"Could not read file: {e}")
else:
    demo_choice = st.sidebar.selectbox("Demo dataset", ["Titanic (classification)", "Tips (regression)"])
    demo_url = (
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
        if demo_choice == "Titanic (classification)"
        else "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
    )
    try:
        df = pd.read_csv(demo_url)
    except Exception:
        st.sidebar.error("Could not fetch demo dataset (no internet). Please upload your own file.")

if df is None:
    st.info("👈 Upload a CSV/Excel file or select a demo dataset from the sidebar to get started.")
    st.stop()

# ---------------------------------------------------------------------------
# Data profile
# ---------------------------------------------------------------------------

st.header("📊 Data Overview")
col1, col2 = st.columns([2, 1])

with col1:
    st.dataframe(df.head(20), use_container_width=True)

with col2:
    profile = profile_dataframe(df)
    st.metric("Rows", profile["n_rows"])
    st.metric("Columns", profile["n_cols"])
    st.metric("Duplicate rows", profile["n_duplicates"])

with st.expander("🔍 Column details & missing values"):
    info_df = pd.DataFrame({
        "dtype": profile["dtypes"],
        "missing_count": profile["missing"],
        "missing_pct": profile["missing_pct"],
    })
    st.dataframe(info_df, use_container_width=True)

if len(profile["numeric_cols"]) > 1:
    with st.expander("📈 Correlation heatmap (numeric columns)"):
        corr = df[profile["numeric_cols"]].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Shared rendering helper -- used by the auto-leaderboard, classification,
# and regression tabs so we don't duplicate the diagnostics/export logic.
# ---------------------------------------------------------------------------

def render_results(results: dict, key_prefix: str):
    """Render leaderboard, diagnostics, feature importance, and export for a
    train_and_evaluate() results dict. key_prefix keeps Streamlit widget keys
    unique across the three tabs."""
    results_df = results["results_df"]
    task = results["task_type"]

    st.subheader("🏆 Results")
    metric_cols = [c for c in results_df.columns if c != "Model"]
    if len(results_df) > 1:
        st.dataframe(
            results_df.style.highlight_max(axis=0, subset=metric_cols, color="lightgreen"),
            use_container_width=True,
        )
    else:
        st.dataframe(results_df, use_container_width=True)

    best_model_name = results_df.iloc[0]["Model"]
    st.success(f"Best model: **{best_model_name}**")
    best_pipeline = results["fitted_pipelines"][best_model_name]

    tab1, tab2, tab3 = st.tabs(["📉 Diagnostics", "⭐ Feature Importance", "💾 Export"])

    with tab1:
        y_test = results["y_test"]
        y_pred = best_pipeline.predict(results["X_test"])

        if task == "classification":
            cm = confusion_matrix(y_test, y_pred)
            labels = sorted(y_test.unique())
            fig = ff.create_annotated_heatmap(
                z=cm, x=[str(l) for l in labels], y=[str(l) for l in labels],
                colorscale="Blues"
            )
            fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_cm")
        else:
            residuals = y_test.values - y_pred
            fig = px.scatter(x=y_pred, y=residuals, labels={"x": "Predicted", "y": "Residual"},
                              title="Residual Plot")
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_resid")

            fig2 = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual", "y": "Predicted"},
                               title="Actual vs Predicted")
            min_v, max_v = float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))
            fig2.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v, line=dict(dash="dash", color="red"))
            st.plotly_chart(fig2, use_container_width=True, key=f"{key_prefix}_actual_pred")

    with tab2:
        fi_df = get_feature_importance(best_pipeline)
        if not fi_df.empty:
            fig = px.bar(fi_df, x="importance", y="feature", orientation="h",
                         title=f"Top features — {best_model_name}")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_fi")
        else:
            st.info("Feature importance is not available for this model type.")

    with tab3:
        model_to_export = st.selectbox(
            "Select model to download:", results_df["Model"].tolist(),
            key=f"{key_prefix}_export_select"
        )
        pipeline_to_export = results["fitted_pipelines"][model_to_export]

        buffer = io.BytesIO()
        joblib.dump(pipeline_to_export, buffer)
        buffer.seek(0)

        st.download_button(
            label=f"⬇️ Download {model_to_export} (.pkl)",
            data=buffer,
            file_name=f"{model_to_export.replace(' ', '_').lower()}_pipeline.pkl",
            mime="application/octet-stream",
            key=f"{key_prefix}_download",
        )
        st.caption("This pickle includes the full preprocessing + model pipeline — ready to use with `pipeline.predict(new_df)`.")


# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

st.header("🎯 Train Models")
main_tab1, main_tab2, main_tab3 = st.tabs([
    "🏆 Auto Leaderboard", "🧮 Classification", "📈 Regression"
])

# --- Tab 1: Auto Leaderboard (trains every model for the detected task) ---
with main_tab1:
    st.caption("Trains every available model for the task and ranks them automatically.")
    target_col_auto = st.selectbox("Target column to predict:", df.columns.tolist(), key="auto_target")

    clean_df_auto = df.dropna(subset=[target_col_auto]).copy()
    task_guess_auto = infer_task_type(clean_df_auto[target_col_auto])
    task_type_auto = st.radio(
        "Task type (auto-detected, override if needed):",
        ["classification", "regression"],
        index=0 if task_guess_auto == "classification" else 1,
        horizontal=True,
        key="auto_task",
    )
    test_size_auto = st.slider("Test set size (%)", 10, 40, 20, key="auto_test_size") / 100

    if st.button("🚀 Train All Models", type="primary", key="auto_train_btn"):
        with st.spinner("Training models... this may take a moment"):
            try:
                st.session_state["results_auto"] = train_and_evaluate(
                    clean_df_auto, target_col_auto, task_type_auto, test_size=test_size_auto
                )
            except Exception as e:
                st.error(f"Training failed: {e}")

    if "results_auto" in st.session_state:
        render_results(st.session_state["results_auto"], key_prefix="auto")

# --- Tab 2: Classification (user picks one specific model) ---
with main_tab2:
    st.caption("Pick a target column, a specific classification model, and a test size.")
    target_col_clf = st.selectbox("Target column to predict:", df.columns.tolist(), key="clf_target")

    clean_df_clf = df.dropna(subset=[target_col_clf]).copy()
    available_clf_models = list(get_models("classification").keys())
    chosen_clf_model = st.selectbox("Model:", available_clf_models, key="clf_model_select")
    test_size_clf = st.slider("Test set size (%)", 10, 40, 20, key="clf_test_size") / 100

    if st.button("🚀 Train Classification Model", type="primary", key="clf_train_btn"):
        with st.spinner(f"Training {chosen_clf_model}..."):
            try:
                st.session_state["results_clf"] = train_and_evaluate(
                    clean_df_clf, target_col_clf, "classification",
                    test_size=test_size_clf, model_names=[chosen_clf_model]
                )
            except Exception as e:
                st.error(f"Training failed: {e}")

    if "results_clf" in st.session_state:
        render_results(st.session_state["results_clf"], key_prefix="clf")

# --- Tab 3: Regression (user picks one specific model) ---
with main_tab3:
    st.caption("Pick a target column, a specific regression model, and a test size.")
    target_col_reg = st.selectbox("Target column to predict:", df.columns.tolist(), key="reg_target")

    clean_df_reg = df.dropna(subset=[target_col_reg]).copy()
    available_reg_models = list(get_models("regression").keys())
    chosen_reg_model = st.selectbox("Model:", available_reg_models, key="reg_model_select")
    test_size_reg = st.slider("Test set size (%)", 10, 40, 20, key="reg_test_size") / 100

    if st.button("🚀 Train Regression Model", type="primary", key="reg_train_btn"):
        with st.spinner(f"Training {chosen_reg_model}..."):
            try:
                st.session_state["results_reg"] = train_and_evaluate(
                    clean_df_reg, target_col_reg, "regression",
                    test_size=test_size_reg, model_names=[chosen_reg_model]
                )
            except Exception as e:
                st.error(f"Training failed: {e}")

    if "results_reg" in st.session_state:
        render_results(st.session_state["results_reg"], key_prefix="reg")
