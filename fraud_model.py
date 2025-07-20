# fraud_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import joblib
import numpy as np
from imblearn.over_sampling import SMOTE # <-- Import SMOTE
import time # Import time for measuring duration

print("--- Starting fraud_model.py script ---")

# --- 1. Load Data ---
start_time = time.time()
try:
    # Ensure 'creditcard.csv' is in the same directory as this script
    df = pd.read_csv('creditcard.csv')
    print(f"creditcard.csv loaded successfully in {time.time() - start_time:.2f} seconds.")
except FileNotFoundError:
    print("Error: 'creditcard.csv' not found. Please download it from Kaggle and place it in the same directory.")
    print("Creating a small DUMMY DATASET for demonstration purposes. **NOTE: This model will not be accurate without real data.**")
    num_samples = 10000 # Increased dummy samples for better simulation
    data = {f'V{i}': np.random.rand(num_samples) * 2 - 1 for i in range(1, 29)}
    data['Time'] = np.random.randint(0, 100000, num_samples)
    data['Amount'] = np.random.rand(num_samples) * 500 + 10
    data['Class'] = np.zeros(num_samples, dtype=int)
    # Simulate some fraud (~0.5% fraud rate for dummy data)
    fraud_indices = np.random.choice(num_samples, int(num_samples * 0.005), replace=False)
    for idx in fraud_indices:
        data['Class'][idx] = 1
        data['Amount'][idx] = np.random.rand() * 1000 + 500 # Higher fraud amounts
        # Adjust some V-features for dummy fraud
        for v in [10, 12, 14, 17]: data[f'V{v}'][idx] = np.random.normal(-5, 2)
        for v in [4, 11, 21]: data[f'V{v}'][idx] = np.random.normal(3, 1)
    df = pd.DataFrame(data)
    print("Using dummy data for demonstration. Training will be faster but less meaningful.")

# --- 2. Preprocessing ---
print("\nStarting data preprocessing...")
X = df.drop(['Class', 'Time'], axis=1) # Drop 'Time' as per common practice; Class is target
y = df['Class']

# Store the column names for later use in prediction (ensures consistent order)
feature_columns = X.columns.tolist()

# Scale the 'Amount' feature
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])
print("Data preprocessing complete (Amount scaled).")

print(f"\nOriginal dataset shape: {df.shape}")
print(f"Original class distribution:\n{y.value_counts()}")

# --- 3. Split Data ---
print("\nSplitting data into training and test sets...")
# Stratify ensures that the proportion of fraud cases is maintained in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data split complete.")

print(f"Training data shape before resampling: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Training class distribution before resampling:\n{y_train.value_counts()}")

# --- 4. Apply SMOTE to the training data ---
print("\nApplying SMOTE to balance the training data. This may take a while...")
smote_start_time = time.time()
smote = SMOTE(random_state=42, sampling_strategy='auto') # 'auto' balances the classes to 50/50
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"SMOTE complete in {time.time() - smote_start_time:.2f} seconds.")

print(f"Training data shape after SMOTE: {X_train_resampled.shape}")
print(f"Training class distribution after SMOTE:\n{y_train_resampled.value_counts()}")

# --- 5. Model Training ---
print("\nTraining RandomForestClassifier. This is the longest step...")
model_train_start_time = time.time()
# Optimized: Reduced n_estimators to 100, and n_jobs to 2
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=2)
model.fit(X_train_resampled, y_train_resampled)
print(f"Model training complete in {time.time() - model_train_start_time:.2f} seconds.")

# --- 6. Evaluation on the original (unresampled) Test Set ---
print("\n--- Evaluating Model Performance on Test Set ---")
eval_start_time = time.time()
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1] # Probability of the positive class (fraud)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Area Under Precision-Recall Curve (AUC-PR) - more informative for imbalanced datasets
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
auc_pr = auc(recall, precision)
print(f"\nArea Under Precision-Recall Curve (AUC-PR): {auc_pr:.4f}")

# --- Find an "Optimal" Threshold ---
# Finds the threshold that maximizes the F1-score for the positive class (fraud).
fscore = (2 * precision * recall) / (precision + recall)
fscore = np.nan_to_num(fscore) # Handle potential NaN values if precision/recall are zero
ix = np.argmax(fscore)

optimal_threshold = thresholds[ix] if ix < len(thresholds) else thresholds[-1] # Fallback for last threshold

print(f'\nOptimal Threshold (maximizing F1-score for fraud): {optimal_threshold:.4f}')
print(f'  Precision at this threshold: {precision[ix]:.4f}')
print(f'  Recall at this threshold: {recall[ix]:.4f}')
print(f'  F1-Score at this threshold: {fscore[ix]:.4f}')
print(f"Model evaluation complete in {time.time() - eval_start_time:.2f} seconds.")


# --- 7. Save Model, Scaler, and Feature Columns ---
print("\nSaving model, scaler, and feature columns...")
joblib.dump(model, 'fraud_detector_model.pkl')
joblib.dump(scaler, 'amount_scaler.pkl')
joblib.dump(feature_columns, 'model_feature_columns.pkl') # Ensures feature order is consistent
print("Model, scaler, and feature columns saved successfully.")
print("\n--- fraud_model.py script finished ---")