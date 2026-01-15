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
        result = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(result)

    df_results = pd.DataFrame(results)
    print("\nClassification Model Evaluation Results:\n")
    print(df_results)


if __name__ == "__main__":
    main()
