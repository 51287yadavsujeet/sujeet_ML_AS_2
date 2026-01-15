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


# ------------------------------
# Model Registry
# ------------------------------
MODEL_REGISTRY = {
    "Logistic Regression": lr_model,
    "Decision Tree": dt_model,
    "KNN": knn_model,
    "Naive Bayes": nb_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}


# ------------------------------
# Evaluation Function
# ------------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = None

    results = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    return results, y_pred


# ------------------------------
# Streamlit UI
# ------------------------------
def main():
    st.set_page_config(page_title="ML Model Evaluation Dashboard", layout="wide")
    st.title("📊 ML Classification Model Evaluation Dashboard")
    st.write("Upload test dataset, select model, and evaluate performance.")

    # ------------------------------
    # Upload Test Dataset
    # ------------------------------
    st.subheader("📁 Upload Test Dataset (CSV)")
    uploaded_file = st.file_uploader("Upload CSV file (Test Data Only)", type=["csv"])

    # ------------------------------
    # Model Selection
    # ------------------------------
    st.subheader("🤖 Select Model")
    selected_model_name = st.selectbox("Choose a Model", list(MODEL_REGISTRY.keys()))

    # ------------------------------
    # Run Evaluation
    # ------------------------------
    if st.button("Run Evaluation"):

        if uploaded_file is None:
            st.error("❌ Please upload a CSV test dataset.")
            return

        with st.spinner("Loading model and evaluating..."):

            # Load test dataset
            test_df = pd.read_csv(uploaded_file)

            if "target" not in test_df.columns:
                st.error("❌ CSV must contain a 'target' column.")
                return

            X_test = test_df.drop("target", axis=1)
            y_test = test_df["target"]

            # Load training dataset and train model
            X_train, _, y_train, _ = load_dataset()

            model = MODEL_REGISTRY[selected_model_name]()
            model.fit(X_train, y_train)

            # Evaluate
            results, y_pred = evaluate_model(model, X_test, y_test)

        # ------------------------------
        # Display Metrics
        # ------------------------------
        st.success("✅ Model evaluation completed successfully!")

        st.subheader("📈 Evaluation Metrics")
        metrics_df = pd.DataFrame(results.items(), columns=["Metric", "Value"])
        st.table(metrics_df)

        # ------------------------------
        # Confusion Matrix
        # ------------------------------
        st.subheader("📊 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"])
        st.dataframe(cm_df)

        # ------------------------------
        # Classification Report
        # ------------------------------
        st.subheader("📄 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        # ------------------------------
        # Download Metrics
        # ------------------------------
        st.subheader("⬇ Download Evaluation Report")
        csv = metrics_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Metrics CSV",
            data=csv,
            file_name="model_evaluation_metrics.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
