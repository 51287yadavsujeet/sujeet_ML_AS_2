import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset():
    """
    Loads UCI Breast Cancer Wisconsin Diagnostic dataset from sklearn,
    scales features and returns train-test split.
    """

    # Load dataset from sklearn (no internet required)
    data = load_breast_cancer()



    # Convert to DataFrame
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target  # 0 = malignant, 1 = benign

    # Split features and labels
    X = df.drop("target", axis=1)
    y = df["target"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
