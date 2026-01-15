import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


def prepare_and_save_test_data():
    # Load dataset
    data = load_breast_cancer()

    # Convert to DataFrame
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    # Split features and labels
    X = df.drop("target", axis=1)
    y = df["target"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y, test_size=0.2, random_state=42, stratify=y
    )

    # Prepare test CSV
    test_df = X_test.copy()
    test_df["target"] = y_test.values

    # Absolute path (change to your folder)
    csv_path = r"C:\SUJEET\python-codes\assigment\breast_cancer_test.csv"

    # Save CSV
    test_df.to_csv(csv_path, index=False)

    print("CSV file created successfully at:")
    print(csv_path)

    # Verify file exists
    print("File exists:", os.path.exists(csv_path))


if __name__ == "__main__":
    prepare_and_save_test_data()
