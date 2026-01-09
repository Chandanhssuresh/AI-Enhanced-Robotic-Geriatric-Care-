import pandas as pd
import numpy as np
import lightgbm as lgb # Import LightGBM
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import LabelEncoder # Keep for potential future use if needed
import seaborn as sns
import matplotlib.pyplot as plt
import time
import joblib # For saving the model
import h5py  # For creating the HDF5 file container

print("Libraries imported successfully.")
print(f"LightGBM version: {lgb.__version__}")
print(f"Joblib version: {joblib.__version__}")
print(f"h5py version: {h5py.__version__}")

# --- Constants ---
# !!! IMPORTANT: Replace 'your_dataset.csv' with the actual path to your file !!!
DATA_FILE_PATH = '/home/chandansuresh/perfect_dataset.csv'
# !!! Choose a name for your saved model file !!!
MODEL_SAVE_PATH = '/mnt/c/Users/chand/LGBM/lgbm_geriatric_health_monitor.h5'
TEST_SIZE = 0.20
RANDOM_STATE = 42

# --- 1. Load Data ---
try:
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    # print("\nDataset Info:")
    # df.info() # Optional: uncomment for detailed info
except FileNotFoundError:
    print(f"Error: The file '{DATA_FILE_PATH}' was not found.")
    print("Please make sure the dataset file is in the correct directory or provide the full path.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    exit()

# --- 2. Data Preprocessing and Feature Engineering ---
print("\n--- Data Preprocessing ---")

# Rename columns for easier access
df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
df = df.rename(columns={
    'HeartRatebpm': 'HeartRate',
    'SpO2Level': 'SpO2',
    'SystolicBloodPressuremmHg': 'SystolicBP',
    'DiastolicBloodPressuremmHg': 'DiastolicBP',
    'BodyTemperatureC': 'BodyTemp',
    'FallDetection': 'FallDetected',
    'DataAccuracy': 'DataAccuracy',
    'HeartRateAlert': 'HR_Alert',
    'SpO2LevelAlert': 'SpO2_Alert',
    'BloodPressureAlert': 'BP_Alert',
    'TemperatureAlert': 'Temp_Alert'
})
print("\nRenamed columns:")
print(df.columns)

# Handle Missing Values (Example: Impute with median for numeric)
print("\nChecking for missing values...")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])
# If there are missing values, uncomment and adapt the imputation below:
numeric_cols_with_na = df.select_dtypes(include=np.number).isnull().any()
cols_to_impute = numeric_cols_with_na[numeric_cols_with_na].index.tolist()
# for col in cols_to_impute:
#     if col in df.columns: # Ensure column exists after renaming
#         median_val = df[col].median()
#         df[col].fillna(median_val, inplace=True)
#         print(f"Imputed missing values in '{col}' with median ({median_val}).")
# Note: Add imputation for categorical columns if needed using mode.


# Define Target Variable: 'Health_Issue_Detected'
# An issue is detected if a fall occurs OR any alert is not 'Normal'
alert_columns = ['HR_Alert', 'SpO2_Alert', 'BP_Alert', 'Temp_Alert']
df['Health_Issue_Detected'] = 0 # Initialize with 0 (No Issue)

# Ensure alert columns exist before using them
existing_alert_columns = [col for col in alert_columns if col in df.columns]
if 'FallDetected' not in df.columns:
    print("Warning: 'FallDetected' column not found after renaming. Check column names.")
    exit()
if not existing_alert_columns:
    print("Warning: None of the specified alert columns found. Check column names.")
    exit()

# Build the condition dynamically based on existing columns
issue_conditions = (df['FallDetected'] == 'Yes')
for col in existing_alert_columns:
    issue_conditions = issue_conditions | (df[col] != 'Normal')

df.loc[issue_conditions, 'Health_Issue_Detected'] = 1

print("\nCreated 'Health_Issue_Detected' target variable.")
target_distribution = df['Health_Issue_Detected'].value_counts(normalize=True) * 100
print("Value counts for 'Health_Issue_Detected':")
print(target_distribution)

# Define Features (X) and Target (y)
# Use the primary sensor readings as features
# Do NOT use the individual alert columns or FallDetected as features (data leakage).
feature_columns = ['HeartRate', 'SpO2', 'SystolicBP', 'DiastolicBP', 'BodyTemp', 'DataAccuracy']
# Verify all feature columns exist
feature_columns = [col for col in feature_columns if col in df.columns]
if len(feature_columns) < 6:
    print(f"Warning: Not all expected feature columns found. Using: {feature_columns}")

X = df[feature_columns]
y = df['Health_Issue_Detected']

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print("\nFeatures used:")
print(feature_columns)

# --- 3. Data Splitting ---
print("\n--- Splitting Data ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print("Class Distribution in Training Set:\n", y_train.value_counts(normalize=True))
print("Class Distribution in Test Set:\n", y_test.value_counts(normalize=True))

# --- 4. Model Training (LightGBM) ---
print("\n--- Training LightGBM Model ---")

# Initialize the LightGBM Classifier
# Parameters are examples, tuning might be needed (e.g., using GridSearchCV)
lgbm_classifier = lgb.LGBMClassifier(
    objective='binary',         # Binary classification
    metric='auc',               # Evaluation metric: Area Under Curve is good for imbalance
    n_estimators=1000,          # Start with a higher number, use early stopping
    learning_rate=0.05,         # Smaller learning rate often requires more estimators
    num_leaves=31,              # Default, balance between complexity and overfitting
    max_depth=-1,               # No limit on depth (num_leaves is often the main control)
    # class_weight='balanced',  # Crucial for imbalanced datasets
    is_unbalance=True,          # Alternative way to handle imbalance in LightGBM
    random_state=RANDOM_STATE,
    n_jobs=-1                   # Use all available CPU cores
    # boosting_type='gbdt',     # Default: Gradient Boosted Decision Trees
    # reg_alpha=0.1,            # L1 regularization (optional)
    # reg_lambda=0.1,           # L2 regularization (optional)
)

start_time = time.time()

# Train with Early Stopping
# Early stopping prevents overfitting and finds a potentially optimal number of trees
# It monitors performance on a validation set (here, we use the test set for simplicity,
# though a separate validation set split from the training data is best practice for final eval)
print("Starting training with early stopping...")
lgbm_classifier.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='auc', # Evaluate based on AUC
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)] # Stop if AUC doesn't improve for 50 rounds
    # Note: verbose=True prints the evaluation results during training
)

end_time = time.time()
print(f"\nModel training completed in {end_time - start_time:.2f} seconds.")
print(f"Best iteration found by early stopping: {lgbm_classifier.best_iteration_}")

# --- 5. Model Evaluation ---
print("\n--- Evaluating Model Performance ---")

# Predict on the Test Set using the best iteration
y_pred = lgbm_classifier.predict(X_test, num_iteration=lgbm_classifier.best_iteration_)
y_pred_proba = lgbm_classifier.predict_proba(X_test, num_iteration=lgbm_classifier.best_iteration_)[:, 1]

# Calculate Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f} (Of predicted issues, how many were real?)")
print(f"Recall:    {recall:.4f} (Of real issues, how many were detected?) <<< Often most important!")
print(f"F1-Score:  {f1:.4f} (Harmonic mean of precision and recall)")
print(f"ROC AUC:   {roc_auc:.4f} (Model's ability to distinguish classes)")

print("\nConfusion Matrix:")
print(conf_matrix)
# Plotting the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Issue', 'Issue Detected'],
            yticklabels=['No Issue', 'Issue Detected'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (LightGBM)')
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Issue', 'Issue Detected']))

# --- 6. Feature Importance ---
print("\n--- Feature Importance (LightGBM) ---")
lgb.plot_importance(lgbm_classifier, figsize=(10, 6), max_num_features=len(feature_columns))
plt.title('Feature Importance from LightGBM')
plt.tight_layout()
plt.show()

# You can also get the raw importance values
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': lgbm_classifier.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importance Values:")
print(importance_df)


# --- 7. Save the Trained Model to H5 file ---
print(f"\n--- Saving Model to {MODEL_SAVE_PATH} ---")
try:
    # We use joblib to serialize the model and h5py to save it within an HDF5 file.
    with h5py.File(MODEL_SAVE_PATH, 'w') as hf:
        # Create a dataset inside the HDF5 file to store the joblib dump
        # joblib.dump needs a file-like object or filename. We give it the h5py dataset object.
        # Note: Saving complex scikit-learn/lightgbm models directly into HDF5 attributes
        # isn't standard. Saving via joblib *into* an HDF5 dataset is a viable way.
        joblib.dump(lgbm_classifier, hf) # Simpler way if joblib handles h5py file objects directly

        # Alternative / More explicit way: Create a dataset first
        # model_dataset = hf.create_dataset('lgbm_model', data=joblib.dumps(lgbm_classifier))
        # print("Model saved as a dataset named 'lgbm_model' within the HDF5 file.")

    print(f"Model successfully saved to: {MODEL_SAVE_PATH}")

    # --- Optional: Example of Loading the Model ---
    print("\n--- Loading Model (Example) ---")
    with h5py.File(MODEL_SAVE_PATH, 'r') as hf:
        loaded_model = joblib.load(hf) # Load directly from file handle

        # If saved using the alternative explicit way:
        # loaded_model = joblib.loads(hf['lgbm_model'][()])

    print("Model loaded successfully.")
    # Verify loaded model by predicting on a sample
    sample_pred = loaded_model.predict(X_test.head(1))
    print(f"Prediction on first test sample using loaded model: {sample_pred}")

except Exception as e:
    print(f"Error saving or loading the model: {e}")


# --- 8. Further Steps & Considerations ---
print("\n--- Further Steps & Considerations ---")
print("1.  **Hyperparameter Tuning:** Use `GridSearchCV` or `RandomizedSearchCV` with `StratifiedKFold` cross-validation on the *training* data to find the optimal hyperparameters for LightGBM (`n_estimators`, `learning_rate`, `num_leaves`, `reg_alpha`, `reg_lambda`, etc.). This is crucial for maximizing performance.")
print("2.  **Cross-Validation:** Implement proper k-fold cross-validation during training/tuning for a more robust evaluation of the model's generalization ability.")
print("3.  **Threshold Adjustment:** Based on the ROC curve or precision-recall curve (using `y_pred_proba`), you might adjust the classification threshold (default is 0.5) to prioritize either higher recall (fewer missed issues) or higher precision (fewer false alarms), depending on the clinical requirements.")
print("4.  **Real-time Pipeline:** For 24/7 monitoring, build a pipeline that takes live sensor data, performs the *exact same* preprocessing steps (renaming, imputation if used), loads the saved model (`.h5` file), and makes predictions.")
print("5.  **Monitoring & Retraining:** Monitor the model's performance in the real world. Data distributions can shift over time (data drift), potentially requiring periodic retraining with new data.")