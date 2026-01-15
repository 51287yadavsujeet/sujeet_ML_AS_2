import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_train_test_csv():
    # Load dataset
    data = load_breast_cancer()

    # Convert to DataFrame
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target  # 0 = Malignant, 1 = Benign

    # Split features and labels
    X = df.drop("target", axis=1)
    y = df["target"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert back to DataFrame
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    final_df = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)

    # Train-Test Split
    train_df, test_df = train_test_split(final_df, test_size=0.2, random_state=42, stratify=y)

    # Save CSV files
    train_df.to_csv("breast_cancer_train.csv", index=False)
    test_df.to_csv("breast_cancer_test.csv", index=False)

    print("Train and Test CSV files generated successfully.")
    print("Train file: breast_cancer_train1.csv")
    print("Test file : breast_cancer_test1.csv")
    print(f"Train rows: {train_df.shape[0]}")
    print(f"Test rows : {test_df.shape[0]}")

if __name__ == "__main__":
    generate_train_test_csv()
