# fraud_model.py - Enhanced Fraud Detection Model with Progress Indicators
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import time
import sys
import os
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings('ignore')

# Progress indicator utilities
class ProgressTracker:
    def __init__(self, total_steps, description="Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        self.step_times = []
    
    def update(self, step_description="", increment=1):
        self.current_step += increment
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if self.current_step > 0:
            avg_time_per_step = elapsed / self.current_step
            remaining_steps = self.total_steps - self.current_step
            estimated_remaining = avg_time_per_step * remaining_steps
            
            # Calculate percentage
            percentage = (self.current_step / self.total_steps) * 100
            
            # Create progress bar
            bar_length = 40
            filled_length = int(bar_length * self.current_step // self.total_steps)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            # Format time
            elapsed_str = self._format_time(elapsed)
            remaining_str = self._format_time(estimated_remaining)
            
            # Clear line and print progress
            sys.stdout.write('\r')
            sys.stdout.write(f'{self.description}: |{bar}| {percentage:5.1f}% | {step_description} | Elapsed: {elapsed_str} | Remaining: {remaining_str}')
            sys.stdout.flush()
            
            if self.current_step >= self.total_steps:
                print()  # New line when complete
    
    def _format_time(self, seconds):
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def complete(self, final_message=""):
        self.update(self.total_steps - self.current_step)
        if final_message:
            print(f"\n✓ {final_message}")

def print_animated_header():
    """Print an animated header for the training process"""
    header = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🚀 Enhanced FraudGuard Model Training 🚀                    ║
║                              with Progress Tracking                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    for line in header.split('\n'):
        print(line)
        time.sleep(0.1)

def print_step_header(step_num, total_steps, title):
    """Print a formatted step header"""
    print(f"\n{'='*80}")
    print(f"📋 STEP {step_num}/{total_steps}: {title}")
    print(f"{'='*80}")

# Initialize progress tracking
print_animated_header()
total_training_steps = 7  # Data loading, feature engineering, data prep, model training, evaluation, saving, summary
progress = ProgressTracker(total_training_steps, "Training Progress")

print("=== Enhanced FraudGuard Model Training ===")

# --- 1. Load and Preprocess Data ---
print_step_header(1, total_training_steps, "Data Loading and Preprocessing")
progress.update("Loading dataset...")

start_time = time.time()
try:
    df = pd.read_csv('creditcard.csv')
    print(f"✓ Real credit card dataset loaded successfully ({df.shape[0]:,} transactions)")
    print(f"✓ Dataset shape: {df.shape}")
    print(f"✓ Fraud rate: {(df['Class'].sum() / len(df) * 100):.3f}%")
    progress.update("Real dataset loaded", 1)
except FileNotFoundError:
    print("⚠️  Real dataset not found. Creating enhanced synthetic dataset...")
    progress.update("Creating synthetic dataset...")
    
    # Create more realistic synthetic data
    num_samples = 50000
    np.random.seed(42)
    
    # Generate realistic transaction features
    data = {}
    
    # Time features (more realistic)
    data['Time'] = np.random.exponential(1000, num_samples)
    progress.update("Generated time features")
    
    # Amount distribution (more realistic with long tail)
    data['Amount'] = np.random.exponential(50, num_samples) + 10
    # Add some high-value transactions
    high_value_indices = np.random.choice(num_samples, int(num_samples * 0.01), replace=False)
    data['Amount'][high_value_indices] = np.random.uniform(1000, 5000, len(high_value_indices))
    progress.update("Generated amount features")
    
    # Generate V-features with realistic correlations
    progress.update("Generating V-features...")
    for i in range(1, 29):
        if i in [1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]:
            data[f'V{i}'] = np.random.normal(0, 1, num_samples)
        else:
            data[f'V{i}'] = np.random.normal(0, 0.5, num_samples)
    
    # Create fraud patterns
    fraud_indices = np.random.choice(num_samples, int(num_samples * 0.0017), replace=False)  # ~0.17% fraud rate
    progress.update("Applying fraud patterns...")
    
    # Apply sophisticated fraud patterns
    for idx in fraud_indices:
        # Amount manipulation for fraud
        data['Amount'][idx] = np.random.choice([
            np.random.uniform(1000, 3000),  # High amount fraud
            np.random.uniform(1, 5)          # Low amount testing
        ])
        
        # V-feature manipulation for fraud detection
        # Strong negative correlations (fraud indicators)
        data['V14'][idx] = np.random.normal(-8, 2)
        data['V12'][idx] = np.random.normal(-6, 2)
        data['V10'][idx] = np.random.normal(-7, 2)
        data['V17'][idx] = np.random.normal(-5, 2)
        
        # Strong positive correlations (fraud indicators)
        data['V4'][idx] = np.random.normal(8, 2)
        data['V11'][idx] = np.random.normal(6, 2)
        data['V21'][idx] = np.random.normal(4, 2)
        
        # Time-based fraud patterns
        data['Time'][idx] = np.random.choice([
            np.random.uniform(0, 1000),      # Early morning
            np.random.uniform(80000, 90000)  # Late night
        ])
    
    data['Class'] = np.zeros(num_samples, dtype=int)
    data['Class'][fraud_indices] = 1
    
    df = pd.DataFrame(data)
    print(f"✓ Synthetic dataset created with {len(fraud_indices)} fraud cases")
    progress.update("Synthetic dataset created", 1)

print(f"✓ Data loading completed in {time.time() - start_time:.2f} seconds")
progress.complete("Data loading completed")

# --- 2. Enhanced Feature Engineering ---
print_step_header(2, total_training_steps, "Feature Engineering")
progress.update("Starting feature engineering...")

feature_start = time.time()

# Store original features
feature_columns = [col for col in df.columns if col not in ['Class', 'Time']]

# Create new engineered features
progress.update("Creating logarithmic features...")
df['Amount_Log'] = np.log1p(df['Amount'])
df['Amount_Squared'] = df['Amount'] ** 2
df['Time_Hour'] = (df['Time'] % 86400) / 3600  # Extract hour of day

# Create interaction features
progress.update("Creating interaction features...")
df['Amount_V14'] = df['Amount'] * df['V14']
df['Amount_V12'] = df['Amount'] * df['V12']
df['V4_V11'] = df['V4'] * df['V11']

# Statistical features
progress.update("Calculating statistical features...")
df['V_Mean'] = df[[f'V{i}' for i in range(1, 29)]].mean(axis=1)
df['V_Std'] = df[[f'V{i}' for i in range(1, 29)]].std(axis=1)
df['V_Max'] = df[[f'V{i}' for i in range(1, 29)]].max(axis=1)
df['V_Min'] = df[[f'V{i}' for i in range(1, 29)]].min(axis=1)

# Risk indicators
progress.update("Creating risk indicators...")
df['High_Amount'] = (df['Amount'] > 1000).astype(int)
df['Low_Amount'] = (df['Amount'] < 10).astype(int)
df['Night_Transaction'] = ((df['Time_Hour'] >= 22) | (df['Time_Hour'] <= 6)).astype(int)

# Update feature columns
feature_columns = [col for col in df.columns if col not in ['Class', 'Time']]

X = df[feature_columns]
y = df['Class']

print(f"✓ Feature engineering completed in {time.time() - feature_start:.2f} seconds")
print(f"✓ Final feature count: {len(feature_columns)}")
progress.complete("Feature engineering completed")

# --- 3. Data Splitting and Scaling ---
print_step_header(3, total_training_steps, "Data Preparation")
progress.update("Splitting data...")

prep_start = time.time()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Use RobustScaler for better handling of outliers
progress.update("Scaling features...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Data preparation completed in {time.time() - prep_start:.2f} seconds")
progress.complete("Data preparation completed")

# --- 4. Enhanced Model Training ---
print_step_header(4, total_training_steps, "Model Training")
progress.update("Initializing models...")

model_start = time.time()

# Try multiple models and select the best one
models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=200, 
        max_depth=15, 
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=8,
        random_state=42
    ),
    'LogisticRegression': LogisticRegression(
        C=1.0,
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    )
}

# Train and evaluate models
best_model = None
best_score = 0
best_model_name = ""

model_names = list(models.keys())
for i, (name, model) in enumerate(models.items()):
    print(f"\n--- Training {name} ({i+1}/{len(models)}) ---")
    progress.update(f"Training {name}...")
    
    # Apply SMOTE for imbalanced data
    progress.update(f"Applying SMOTE for {name}...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    # Train model
    progress.update(f"Fitting {name} model...")
    model.fit(X_train_resampled, y_train_resampled)
    
    # Predict on test set
    progress.update(f"Evaluating {name}...")
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)
    
    print(f"✓ {name} Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  AUC-ROC: {auc_score:.4f}")
    
    # Select best model based on F1-score (important for imbalanced data)
    if f1 > best_score:
        best_score = f1
        best_model = model
        best_model_name = name

print(f"\n✓ Best model: {best_model_name} (F1-Score: {best_score:.4f})")
progress.complete("Model training completed")

# --- 5. Final Model Evaluation ---
print_step_header(5, total_training_steps, "Model Evaluation")
progress.update("Evaluating best model...")

eval_start = time.time()

# Retrain best model with full training data
progress.update("Retraining best model...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
best_model.fit(X_train_resampled, y_train_resampled)

# Final predictions
progress.update("Making final predictions...")
y_pred_final = best_model.predict(X_test_scaled)
y_proba_final = best_model.predict_proba(X_test_scaled)[:, 1]

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_final))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_final))

# Calculate additional metrics
progress.update("Calculating advanced metrics...")
precision, recall, thresholds = precision_recall_curve(y_test, y_proba_final)
auc_pr = auc(recall, precision)
roc_auc = roc_auc_score(y_test, y_proba_final)

print(f"\nAdvanced Metrics:")
print(f"✓ Area Under Precision-Recall Curve: {auc_pr:.4f}")
print(f"✓ Area Under ROC Curve: {roc_auc:.4f}")

# Find optimal threshold
progress.update("Finding optimal threshold...")
fscore = (2 * precision * recall) / (precision + recall)
fscore = np.nan_to_num(fscore)
ix = np.argmax(fscore)
optimal_threshold = thresholds[ix] if ix < len(thresholds) else thresholds[-1]

print(f"\nOptimal Threshold: {optimal_threshold:.4f}")
print(f"  Precision at optimal threshold: {precision[ix]:.4f}")
print(f"  Recall at optimal threshold: {recall[ix]:.4f}")
print(f"  F1-Score at optimal threshold: {fscore[ix]:.4f}")

print(f"\n✓ Model evaluation completed in {time.time() - eval_start:.2f} seconds")
progress.complete("Model evaluation completed")

# --- 6. Save Enhanced Model ---
print_step_header(6, total_training_steps, "Saving Model Assets")
progress.update("Saving model...")

save_start = time.time()

# Save the best model
joblib.dump(best_model, 'fraud_detector_model.pkl')
progress.update("Model saved")

# Save the scaler
joblib.dump(scaler, 'amount_scaler.pkl')
progress.update("Scaler saved")

# Save feature columns (including engineered features)
joblib.dump(feature_columns, 'model_feature_columns.pkl')
progress.update("Feature columns saved")

# Save optimal threshold
joblib.dump(optimal_threshold, 'optimal_threshold.pkl')
progress.update("Optimal threshold saved")

print(f"✓ Model assets saved successfully in {time.time() - save_start:.2f} seconds")
progress.complete("Model assets saved")

# --- 7. Model Performance Summary ---
print_step_header(7, total_training_steps, "Training Summary")
progress.update("Generating summary...")

print("\n=== Model Performance Summary ===")
print(f"✓ Model Type: {best_model_name}")
print(f"✓ Total Features: {len(feature_columns)}")
print(f"✓ Training Samples: {len(X_train)}")
print(f"✓ Test Samples: {len(X_test)}")
print(f"✓ Fraud Rate in Dataset: {(y.sum() / len(y) * 100):.3f}%")
print(f"✓ Best F1-Score: {best_score:.4f}")
print(f"✓ AUC-ROC: {roc_auc:.4f}")
print(f"✓ AUC-PR: {auc_pr:.4f}")
print(f"✓ Optimal Threshold: {optimal_threshold:.4f}")

total_time = time.time() - start_time
print(f"\n=== Enhanced FraudGuard Model Training Completed ===")
print(f"✓ Total execution time: {total_time:.2f} seconds")

# Final animated completion message
completion_message = f"""
🎉 Training Complete! 🎉

╔══════════════════════════════════════════════════════════════════════════════╗
║                              🏆 SUCCESS! 🏆                                  ║
║                                                                              ║
║  • Model: {best_model_name:<15} • F1-Score: {best_score:.4f}                    ║
║  • Features: {len(feature_columns):<3} • Training Time: {total_time:.1f}s              ║
║  • AUC-ROC: {roc_auc:.4f} • Optimal Threshold: {optimal_threshold:.4f}              ║
║                                                                              ║
║  Your fraud detection model is ready to use! 🚀                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
print(completion_message)
progress.complete("Training completed successfully!")