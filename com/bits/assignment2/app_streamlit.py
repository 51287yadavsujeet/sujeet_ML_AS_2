import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

from model.dataset_loader import load_dataset
from model.logistic_regression import get_model as lr_model
from model.decision_tree import get_model as dt_model
from model.knn import get_model as knn_model
from model.naive_bayes import get_model as nb_model
from model.random_forest import get_model as rf_model
from model.xgboost_model import get_model as xgb_model


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    return results


def main():
    st.set_page_config(page_title="ML Model Evaluation Dashboard", layout="wide")

    st.title("📊 Classification Model Evaluation Dashboard")
    st.write("Evaluate multiple machine learning models on the same dataset.")

    if st.button("Run Model Evaluation"):
        with st.spinner("Loading dataset and training models..."):
            X_train, X_test, y_train, y_test = load_dataset()

            models = [
                ("Logistic Regression", lr_model()),
                ("Decision Tree", dt_model()),
                ("KNN", knn_model()),
                ("Naive Bayes", nb_model()),
                ("Random Forest", rf_model()),
                ("XGBoost", xgb_model())
            ]

            results = []

            for name, model in models:
                st.write(f"Training **{name}**...")
                result = evaluate_model(name, model, X_train, X_test, y_train, y_test)
                results.append(result)

            df_results = pd.DataFrame(results)

        st.success("Model evaluation completed successfully!")
        st.subheader("📈 Evaluation Results")
        st.dataframe(df_results, use_container_width=True)

        st.subheader("⬇ Download Results")
        csv = df_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="model_evaluation_results.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
