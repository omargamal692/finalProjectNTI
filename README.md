Fraud Detection Using Machine Learning
Overview

This project focuses on detecting fraudulent financial transactions using machine learning techniques.

The goal is to build a binary classification model that predicts whether a transaction is fraudulent or legitimate based on transaction-related features.

Target Variable:

is_fraud
0 → Legitimate Transaction
1 → Fraudulent Transaction
Dataset

Dataset Source:

Kaggle Fraud Detection Dataset

Dataset Characteristics:

7,000 transaction records
12 input features
1 target feature (is_fraud)
Contains missing values that were handled during preprocessing
Features
transaction_amount
hour_of_day
is_weekend
num_items
customer_age
prev_transactions
distance_from_home
device_type
network_quality
is_first_transaction
store_type
velocity_score
Data Preprocessing
Missing values analysis
Median imputation for numerical features
Exploratory Data Analysis (EDA)
Correlation analysis
Train-Test Split (80%-20%)
Exploratory Data Analysis

The analysis included:

Missing values visualization
Fraud distribution analysis
Transaction amount distribution
Correlation heatmap
Models Used
Logistic Regression

Accuracy: 89.71%

Decision Tree Classifier

Accuracy: 89.71%

Random Forest Classifier

Accuracy: 89.71%

Technologies
Python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-Learn
Project Structure
Fraud-Detection/
│
├── data/
│   └── fraud.csv
│
├── notebooks/
│   └── Fraud_Detection.ipynb
│
├── images/
│   ├── missing_values.png
│   ├── fraud_distribution.png
│   └── correlation_heatmap.png
│
├── README.md
└── requirements.txt
Results

The models achieved an accuracy of approximately 89.7% in identifying fraudulent and legitimate transactions.

This project demonstrates a complete machine learning workflow including data preprocessing, exploratory data analysis, model training, and evaluation for fraud detection.
