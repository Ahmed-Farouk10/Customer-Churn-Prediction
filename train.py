# src/train.py
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

# Load preprocessed data
X_train = pd.read_csv('data/X_train_balanced.csv')
y_train = pd.read_csv('data/y_train_balanced.csv')['Churn']
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')['Churn']

# Define base models
base_models = {
    'logistic_regression': LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    ),
    'random_forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    ),
    'svm': LinearSVC(
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    ),
    'knn': KNeighborsClassifier(
        n_neighbors=5
    ),
    'naive_bayes': GaussianNB(),
    'xgboost': XGBClassifier(
        scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),
        max_depth=5,
        learning_rate=0.1,
        n_estimators=100,
        eval_metric='aucpr',
        use_label_encoder=False,
        random_state=42
    )
}

# Create calibrated models
models = {}
for name, model in base_models.items():
    print(f"Training {name}...")
    if name in ['svm', 'knn']:  # These models need calibration
        calibrated_model = CalibratedClassifierCV(model, cv=5)
        calibrated_model.fit(X_train, y_train)
        models[name] = calibrated_model
    else:
        model.fit(X_train, y_train)
        models[name] = model
    print(f"{name} training completed.")

# Save individual models
for name, model in models.items():
    with open(f'models/{name}_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"{name} model saved to models/{name}_model.pkl")

# Create ensemble model with soft voting
ensemble_model = VotingClassifier(
    estimators=[
        ('logistic_regression', models['logistic_regression']),
        ('random_forest', models['random_forest']),
        ('svm', models['svm']),
        ('knn', models['knn']),
        ('naive_bayes', models['naive_bayes']),
        ('xgboost', models['xgboost'])
    ],
    voting='soft'
)

# Train the ensemble model
print("Training ensemble model...")
ensemble_model.fit(X_train, y_train)

# Save the ensemble model
with open('models/ensemble_model.pkl', 'wb') as f:
    pickle.dump(ensemble_model, f)
print("Ensemble model saved to models/ensemble_model.pkl")

# Save all models together for evaluation
with open('models/all_models.pkl', 'wb') as f:
    pickle.dump({
        'individual_models': models,
        'ensemble_model': ensemble_model
    }, f)
print("All models saved to models/all_models.pkl")