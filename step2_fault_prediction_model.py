import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Working directory configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# 2. Ingest baseline dataset
dataset_path = 'telecom_dataset.csv'
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset '{dataset_path}' not found. Please run step1_network_telemetry.py first.")

df = pd.read_csv(dataset_path)

# 3. Target Label Engineering
# Binary label: 1 (Critical Failure) if Drop Call Rate > 3.0% or SNR < 10.0 dB; 0 (Normal) otherwise
df['Is_Critical'] = np.where((df['Drop_Call_Rate_Pct'] > 3.0) | (df['SNR_dB'] < 10.0), 1, 0)

# Feature and Target Definition
X = df[['Connected_Users', 'SNR_dB', 'Latency_ms']]
y = df['Is_Critical']

# 4. Stratified Train / Test Partition (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("--- Data Split Summary ---")
print(f"Total Telemetry Records : {len(df)}")
print(f"Training Partition      : {len(X_train)} records")
print(f"Testing Partition       : {len(X_test)} records")
print("\nTarget Class Distribution (0: Normal, 1: Critical):")
print(y.value_counts())

# 5. Model Training (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluation on Unseen Test Partition
y_pred = model.predict(X_test)

print("\n--- Model Evaluation Results ---")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Feature Importance Analysis
feature_importances = model.feature_importances_
features = X.columns

print("\n--- Feature Importance Coefficients ---")
for feature, importance in zip(features, feature_importances):
    print(f"{feature:20s}: {importance:.4f}")

# 8. Feature Importance Visualization
plt.figure(figsize=(8, 5))
plt.bar(features, feature_importances, color=['#4C72B0', '#C44E52', '#55A868'])
plt.title('Impact of Network Telemetry Metrics on Fault Classification', fontsize=12, fontweight='bold')
plt.xlabel('Telemetry Features')
plt.ylabel('Importance Score (0.0 to 1.0)')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plot_file = 'feature_importance_plot.png'
plt.savefig(plot_file)
plt.close()

# 9. Model Serialization
model_file = 'telecom_rf_model.pkl'
joblib.dump(model, model_file)
print(f"\n[SUCCESS] Model weights serialized to '{model_file}'.")
print(f"[SUCCESS] Feature importance plot saved to '{plot_file}'.")
