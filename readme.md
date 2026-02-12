Name: Sujeet Kumar Yadav
BITS ID: 2025AA05326

This is readme file: Created by Sujeet Kumar Yadav for ML2-Assignment.
#####
**a:Problem statement:**
Objective of this assignment is to develop and evaluate machine learning classification models to predict whether a breast tumor is malignant or benign using the UCI Breast Cancer dataset. 
Dataset contains features computed from digitized images of fine needle aspirate (FNA) of breast masses.
Multiple classification algorithms are applied and compared using evaluation metrics such as Accuracy, AUC, Precision, Recall, F1-Score, and MCC to determine the most effective model for accurate breast cancer diagnosis.

**b. Dataset description:**
Dataset used her is the Breast Cancer Wisconsin (Diagnostic) dataset, obtained from the UCI Machine Learning Repository and available in the sklearn.datasets module. 
This is used for binary classification problems in medical diagnosis.
1. Dataset Overview
Total number of samples: 569
Number of features: 30 numerical features
Target classes: 2 (Binary Classification)
0 → Malignant (cancerous)
1 → Benign (non-cancerous)
2. Feature Description:
30 features are grouped into three categories:
Mean values (mean radius, mean texture, mean perimeter)
Standard error values
Worst (largest) values

**c. Models used and Comparison:**
| **ML Model Name**            | **Accuracy** | **AUC** | **Precision** | **Recall** | **F1 Score** | **MCC** |
| ---------------------------- | ------------ | ------- | ------------- | ---------- | ------------ | ------- |
| Logistic Regression          | 0.9825       | 0.9954  | 0.9861        | 0.9861     | 0.9861       | 0.9623  |
| Decision Tree                | 0.9123       | 0.9157  | 0.9559        | 0.9028     | 0.9286       | 0.8174  |
| kNN                          | 0.9649       | 0.9792  | 0.9595        | 0.9861     | 0.9726       | 0.9245  |
| Naive Bayes                  | 0.9298       | 0.9868  | 0.9444        | 0.9444     | 0.9444       | 0.8492  |
| Random Forest (Ensemble)     | 0.9561       | 0.9937  | 0.9589        | 0.9722     | 0.9655       | 0.9054  |
| XGBoost (Ensemble)           | 0.9561       | 0.9901  | 0.9467        | 0.9861     | 0.9660       | 0.9058  |


**Observations on the performance of each model on the chosen dataset.** 

| **ML Model Name**            | **Observation about model performance**                                                                                                                                                                          |
----------------------------------------------------------------------------------
|Logistic Regression | Logistic Regression showed the best overall performance with very high accuracy (98.25%), AUC (0.9954), and MCC (0.9623). This indicates excellent class separation and reliable predictions on the dataset.
|Decision Tree       | Decision Tree achieved moderate performance with lower accuracy and MCC compared to other models. above results suggest possible overfitting and less stability on unseen test data.
|kNN                 | kNN performed very well with high accuracy and recall, showing that it effectively classified nearby data points. However, it was slightly less accurate than Logistic Regression.
|Naive Bayes         | Naive Bayes delivered good baseline performance with reasonable accuracy and AUC. Its performance is limited by the assumption of feature independence.
|Random Forest (Ensemble) | Random Forest showed strong and stable performance** with high accuracy, F1-score, and MCC. The ensemble approach helped reduce overfitting compared to a single decision tree.
|XGBoost (Ensemble)       | XGBoost achieved excellent performance, close to Random Forest, with high recall and F1-score. It effectively learned complex patterns in the data, making it one of the top-performing models.


