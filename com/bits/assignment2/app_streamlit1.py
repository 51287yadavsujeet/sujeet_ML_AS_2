import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

from model.dataset_loader import load_dataset
from model.logistic_regression import get_model as lr_model
from model.decision_tree import get_model as dt_model
from model.knn import get_model as knn_model
from model.naive_bayes import get_model as nb_model
from model.random_forest import get_model as rf_model
from model.xgboost_model import get_model as xgb_model


# ------------------ Model Registry ------------------
MODEL_REGISTRY = {
    "Logistic Regression": lr_model,
    "Decision Tree": dt_model,
    "KNN": knn_model,
    "Naive Bayes": nb_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}


# ------------------ Evaluation ------------------
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return metrics, cm, report


# ------------------ Streamlit App ------------------
def main():
    st.set_page_config("ML Classification Dashboard", layout="wide")
    st.title("📊 Machine Learning Classification Evaluation")

    # ------------------ Sidebar ------------------
    st.sidebar.header("⚙ Configuration")

    model_name = st.sidebar.selectbox(
        "Select Model",
        options=list(MODEL_REGISTRY.keys())
    )

    uploaded_file = st.sidebar.file_uploader(
        "Upload Test Dataset (CSV Only)",
        type=["csv"],
        help="Upload ONLY test data due to Streamlit free tier limits"
    )

    run_btn = st.sidebar.button("Run Evaluation")

    # ------------------ Load Training Data ------------------
    X_train, X_test_default, y_train, y_test_default = load_dataset()

    # ------------------ Handle Uploaded Test Data ------------------
    if uploaded_file:
        st.info("Using uploaded CSV file as TEST dataset")

        df_test = pd.read_csv(uploaded_file)

        X_test = df_test.drop("diagnosis", axis=1)
        y_test = df_test["diagnosis"].map({"M": 1, "B": 0})
    else:
        X_test, y_test = X_test_default, y_test_default

    # ------------------ Run Model ------------------
    if run_btn:
        st.subheader(f"🔍 Model Selected: {model_name}")

        model = MODEL_REGISTRY[model_name]()

        with st.spinner("Training and evaluating model..."):
            metrics, cm, report = evaluate_model(
                model, X_train, X_test, y_train, y_test
            )

        # ------------------ Metrics Display ------------------
        st.subheader("📈 Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        col1.metric("Precision", f"{metrics['Precision']:.4f}")

        col2.metric("Recall", f"{metrics['Recall']:.4f}")
        col2.metric("F1 Score", f"{metrics['F1 Score']:.4f}")

        col3.metric("AUC", f"{metrics['AUC']:.4f}")
        col3.metric("MCC", f"{metrics['MCC']:.4f}")

        # ------------------ Confusion Matrix ------------------
        st.subheader("🧮 Confusion Matrix")
        cm_df = pd.DataFrame(
            cm,
            columns=["Predicted Negative", "Predicted Positive"],
            index=["Actual Negative", "Actual Positive"]
        )
        st.dataframe(cm_df, use_container_width=True)

        # ------------------ Classification Report ------------------
        st.subheader("📋 Classification Report")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)

        # ------------------ Download Metrics ------------------
        st.subheader("⬇ Download Results")

        metrics_df = pd.DataFrame([metrics])
        csv = metrics_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Metrics CSV",
            data=csv,
            file_name=f"{model_name}_metrics.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
