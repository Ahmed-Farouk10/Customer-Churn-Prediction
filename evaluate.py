# src/evaluate.py
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score
)
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load test data
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')['Churn']

# Load all models
with open('models/all_models.pkl', 'rb') as f:
    models = pickle.load(f)

# Function to evaluate model
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_pred_proba),
        'Average Precision': average_precision_score(y_test, y_pred_proba)
    }
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'results/{model_name}_confusion_matrix.png')
    plt.close()
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {metrics["ROC AUC"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.savefig(f'results/{model_name}_roc_curve.png')
    plt.close()
    
    # Plot Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR curve (AP = {metrics["Average Precision"]:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.savefig(f'results/{model_name}_pr_curve.png')
    plt.close()
    
    return metrics

# Evaluate individual models
results = []
for name, model in models['individual_models'].items():
    print(f"\nEvaluating {name}...")
    metrics = evaluate_model(model, X_test, y_test, name)
    results.append(metrics)

# Evaluate ensemble model
print("\nEvaluating Ensemble Model...")
ensemble_metrics = evaluate_model(models['ensemble_model'], X_test, y_test, 'Ensemble')
results.append(ensemble_metrics)

# Create results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.round(4)

# Save results
results_df.to_csv('results/model_evaluation_results.csv', index=False)
print("\nResults saved to results/model_evaluation_results.csv")

# Print results
print("\nModel Evaluation Results:")
print(results_df.to_string(index=False))

# Plot comparison of all models
plt.figure(figsize=(12, 6))
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Average Precision']
results_df.set_index('Model')[metrics_to_plot].plot(kind='bar')
plt.title('Model Performance Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/model_comparison.png')
plt.close()

# Create a summary of the best model for each metric
best_models = {}
for metric in metrics_to_plot:
    best_model = results_df.loc[results_df[metric].idxmax()]
    best_models[metric] = {
        'Model': best_model['Model'],
        'Score': best_model[metric]
    }

print("\nBest Model for Each Metric:")
for metric, info in best_models.items():
    print(f"{metric}: {info['Model']} ({info['Score']:.4f})")