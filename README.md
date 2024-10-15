# Credit Score Prediction
# Introduction
Predicting credit scores is crucial for financial institutions to assess the creditworthiness of individuals. This project applies machine learning techniques to classify credit scores based on a variety of demographic, financial, and behavioral features. The goal is to build a model that can accurately predict a customer's credit score, helping institutions make informed decisions regarding loans and other financial services.

# Dataset
The dataset used for this project is publicly available on Kaggle: "(https://www.kaggle.com/datasets/parisrohan/credit-score-classification)"

The key features of the dataset include:

ID: Unique identifier for each record.
Customer_ID: Unique identifier for each customer.
Month: Month of the record.
SSN: Social Security Number of the customer.
Amount_invested_monthly: Amount invested monthly by the customer.
Annual_Income, Age, Payment_Behaviour, Credit_History_Age, and several other features providing demographic information, payment history, and credit behavior.
# Preprocessing
The following steps were taken to clean and preprocess the data:
Handling Missing Values: Missing values were handled using mean imputation for numerical features.
Label Encoding: Categorical variables such as 'Credit_Score' and 'Occupation' were label-encoded to transform them into numerical values.
Feature Engineering: Mean encoding was applied to some numerical features such as 'Annual_Income' and 'Outstanding_Debt' based on the 'Credit_Score'.
Dropping Irrelevant Columns: Irrelevant features such as 'ID', 'Customer_ID', 'Month', and 'SSN' were removed to simplify the dataset and avoid overfitting.

# Model Building
Three machine learning models were trained and evaluated for this classification task:
1. Logistic Regression: A simple and interpretable model for binary and multi-class classification.
2. K-Nearest Neighbors (KNN): A distance-based classification algorithm that makes predictions based on the 'k' nearest data points in the training set.
3. Linear Support Vector Classifier (LinearSVC): A linear version of the support vector machine that finds the optimal hyperplane to separate classes.
# Evaluation
Each model was evaluated on both the training and testing datasets using accuracy score. The results are as follows:
1. Logistic Regression:
Training Accuracy: ~98%
Testing Accuracy: ~98%

2. K-Nearest Neighbors (KNN):
Training Accuracy: ~80%
Testing Accuracy: ~67%

3. LinearSVC:
Training Accuracy: ~98%
Testing Accuracy: ~98%

The LinearSVC model provided the best performance with an accuracy of approximately 98% on both the training and test sets.

# Confusion Matrix and Classification Report
For a deeper evaluation, a confusion matrix and classification report were generated for the LinearSVC model, showing precision, recall, F1-score, and support for each class.

# Model Deployment
The LinearSVC model and the data scaler were saved using joblib to allow for future deployment and predictions. This enables the model to be reused without retraining, making it efficient for practical applications.

# Predictions
After training, predictions were made on the test data, and the results were saved to a CSV file named creditworthiness_predictions.csv. This file contains both the predicted creditworthiness class and the actual class for comparison.

# Conclusion
This project demonstrates how machine learning models, specifically Logistic Regression, KNN, and LinearSVC, can be applied to predict credit scores effectively. The LinearSVC model showed the highest accuracy and generalizability, making it a suitable candidate for real-world credit score prediction.
