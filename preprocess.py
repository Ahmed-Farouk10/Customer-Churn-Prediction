# src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import pickle
import os

# Define paths for saving preprocessing objects
scaler_path = 'data/scaler.pkl'
label_encoders_path = 'data/label_encoders.pkl'

# Create the data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Load data
train_df = pd.read_csv('data/customer_churn_dataset-training-master.csv')
test_df = pd.read_csv('data/customer_churn_dataset-testing-master.csv')

# Drop CustomerID (not useful for prediction)
train_df = train_df.drop('CustomerID', axis=1)
test_df = test_df.drop('CustomerID', axis=1)

# Handle missing values in target variable
train_df = train_df.dropna(subset=['Churn'])
test_df = test_df.dropna(subset=['Churn'])

# Separate target variable before handling categorical features and scaling
y_train = train_df['Churn']
X_train = train_df.drop('Churn', axis=1)
y_test = test_df['Churn']
X_test = test_df.drop('Churn', axis=1)

# Calculate derived features
X_train['Support_Calls_per_Tenure'] = X_train['Support Calls'] / X_train['Tenure'].replace(0, 1e-6)
X_train['Payment_Delay_to_Spend'] = X_train['Payment Delay'] / X_train['Total Spend'].replace(0, 1e-6)
X_test['Support_Calls_per_Tenure'] = X_test['Support Calls'] / X_test['Tenure'].replace(0, 1e-6)
X_test['Payment_Delay_to_Spend'] = X_test['Payment Delay'] / X_test['Total Spend'].replace(0, 1e-6)

# Handle categorical variables
categorical_cols = ['Gender', 'Subscription Type', 'Contract Length']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined_series = pd.concat([X_train[col], X_test[col]], axis=0).astype(str).unique()
    le.fit(combined_series)
    X_train[col] = le.transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

# Save label encoders
with open(label_encoders_path, 'wb') as f:
    pickle.dump(label_encoders, f)

# Identify numerical columns, including derived features
numerical_cols = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction', 'Support_Calls_per_Tenure', 'Payment_Delay_to_Spend']

# Initialize and fit StandardScaler on all numerical features, including derived ones
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Save the scaler
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

# Handle imbalanced data with SMOTE *after* scaling
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Save preprocessed data
pd.DataFrame(X_train_balanced, columns=X_train.columns).to_csv('data/X_train_balanced.csv', index=False)
pd.Series(y_train_balanced, name='Churn').to_csv('data/y_train_balanced.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
pd.Series(y_test, name='Churn').to_csv('data/y_test.csv', index=False)

print("Preprocessing completed. Balanced training data, test data, scaler, and label encoders saved.")