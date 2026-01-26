import streamlit as st
import pandas as pd
import os

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

# ---------------- Model Registry ----------------
MODEL_REGISTRY = {
    "Logistic Regression": lr_model,
    "Decision Tree": dt_model,
    "KNN": knn_model,
    "Naive Bayes": nb_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

LOCAL_TEST_FILE = "breast_cancer_test.csv"


# ---------------- Evaluation Function ----------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    auc = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

    results = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    return results, y_pred


# ---------------- Streamlit App UI Level Set up ----------------
def main():
    st.set_page_config(page_title="ML Model Dashboard", layout="wide")
    st.title("ML Classification Model Dashboard")

    # -------- Download Sample Test File (FIRST) --------
    st.subheader("First Download Sample Test Dataset")

    if os.path.exists(LOCAL_TEST_FILE):
        with open(LOCAL_TEST_FILE, "rb") as f:
            st.download_button(
                label="Download Sample Test File (breast_cancer_test.csv)",
                data=f,
                file_name=LOCAL_TEST_FILE,
                mime="text/csv"
            )
    else:
        st.warning("Sample test file not found in application folder.")

    # -------- Upload the csv file only --------
    st.subheader("Upload Test Dataset (.csv only)")
    uploaded_file = st.file_uploader(
        "Upload CSV file (Test Data Only)",
        type=["csv"]
    )

    # -------- Model Selection --------
    st.subheader("Choose the Model")
    selected_model_name = st.selectbox(
        "Select Model",
        list(MODEL_REGISTRY.keys())
    )

    # -------- Run Evaluation --------
    if st.button("Run Evaluation"):

        with st.spinner("Loading data and evaluating model..."):

            # Load Test Data
            if uploaded_file is not None:
                test_df = pd.read_csv(uploaded_file)
                st.info("Using uploaded test dataset.")
            else:
                if not os.path.exists(LOCAL_TEST_FILE):
                    st.error("No uploaded file and default test file not found.")
                    return
                test_df = pd.read_csv(LOCAL_TEST_FILE)
                st.info("Using default breast_cancer_test.csv from local folder.")

            if "target" not in test_df.columns:
                st.error("CSV must contain a 'target' column.")
                return

            X_test = test_df.drop("target", axis=1)
            y_test = test_df["target"]

            # Train Model
            X_train, _, y_train, _ = load_dataset()
            model = MODEL_REGISTRY[selected_model_name]()
            model.fit(X_train, y_train)

            # Evaluate
            results, y_pred = evaluate_model(model, X_test, y_test)

        # -------- Results --------
        st.success("Model evaluation completed successfully!")

        st.subheader("Evaluation Metrics")
        metrics_df = pd.DataFrame(results.items(), columns=["Metric", "Value"])
        st.table(metrics_df)

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(
            cm,
            columns=["Predicted 0", "Predicted 1"],
            index=["Actual 0", "Actual 1"]
        )
        st.dataframe(cm_df)


if __name__ == "__main__":
    main()
    # -------- Footer --------
    st.markdown(
        """
        <hr>
        <div style="text-align: center; color: grey; font-size: 14px;">
            Created by <b>Sujeet Kumar Yadav</b> | BITS ID: 2025AA05326
        </div>
        """,
        unsafe_allow_html=True
    )
