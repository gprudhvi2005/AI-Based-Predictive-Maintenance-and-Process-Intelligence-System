import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ==========================================
# 1. Load Preprocessed Datasets
# ==========================================
# Read the CSV files saved in your Milestone_1 directory
X_train_scaled = pd.read_csv('X_train_scaled.csv')
X_test_scaled = pd.read_csv('X_test_scaled.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

# ==========================================
# 2. Initialize Model & Fit on Training Data
# ==========================================
model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# ==========================================
# 3. Predict on Test Data
# ==========================================
y_pred = model.predict(X_test_scaled)                 # Default 0.5 threshold
y_proba = model.predict_proba(X_test_scaled)[:, 1]     # Failure probability scores

# ==========================================
# 4. Print Confusion Matrix & Metrics
# ==========================================
print("=== Confusion Matrix (Threshold 0.50) ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, zero_division=0))

# ROC-AUC score requires both classes (0 and 1) in y_test
if len(np.unique(y_test)) > 1:
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# ==========================================
# 5. Evaluate with Custom Threshold (0.30)
# ==========================================
custom_threshold = 0.30
y_pred_custom = (y_proba >= custom_threshold).astype(int)

print(f"\n=== Results with Custom Threshold ({custom_threshold}) ===")
print(confusion_matrix(y_test, y_pred_custom))
print(classification_report(y_test, y_pred_custom, zero_division=0))

# ==========================================
# 6. Rank High-Risk Maintenance Predictions
# ==========================================
results_df = pd.DataFrame({
    'Actual Label': y_test,
    'Predicted Label': y_pred_custom,
    'Failure Risk (%)': np.round(y_proba * 100, 2)
})

print("\n=== Top 10 Highest Risk Predictions ===")
print(results_df.sort_values(by='Failure Risk (%)', ascending=False).head(10).to_string(index=False))