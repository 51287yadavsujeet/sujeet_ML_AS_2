import pandas as pd
from sklearn.datasets import load_breast_cancer

def save_dataset_to_csv():
    # Load dataset from sklearn
    data = load_breast_cancer()

    # Convert to DataFrame
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target  # 0 = malignant, 1 = benign

    # Save to CSV
    file_name = "breast_cancer_dataset.csv"
    df.to_csv(file_name, index=False)

    print(f"Dataset successfully saved as: {file_name}")
    print(f"Total Rows: {df.shape[0]}")
    print(f"Total Columns: {df.shape[1]}")

if __name__ == "__main__":
    save_dataset_to_csv()
