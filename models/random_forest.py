import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import time # To measure training time

print("Libraries imported successfully.")

# --- 1. Load Data ---
try:
    # !!! IMPORTANT: Replace 'your_dataset.csv' with the actual path to your file !!!
    file_path = '/home/chandansuresh/perfect_dataset.csv'
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    print("\nDataset Info:")
    df.info()
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    print("Please make sure the dataset file is in the correct directory or provide the full path.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    exit()

# --- 2. Data Preprocessing and Feature Engineering ---
print("\n--- Data Preprocessing ---")

# Rename columns for easier access (remove spaces, special chars)
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

# Handle Missing Values (Example: Impute with median for numeric, mode for categoric)
print("\nChecking for missing values...")
print(df.isnull().sum())
# If there are missing values, uncomment and adapt the imputation below:
# numeric_cols = df.select_dtypes(include=np.number).columns
# categorical_cols = df.select_dtypes(include='object').columns
# for col in numeric_cols:
#     if df[col].isnull().any():
#         median_val = df[col].median()
#         df[col].fillna(median_val, inplace=True)
#         print(f"Imputed missing values in '{col}' with median ({median_val}).")
# for col in categorical_cols:
#      if df[col].isnull().any():
#         mode_val = df[col].mode()[0]
#         df[col].fillna(mode_val, inplace=True)
#         print(f"Imputed missing values in '{col}' with mode ('{mode_val}').")


# Define Target Variable: 'Health_Issue_Detected'
# An issue is detected if a fall occurs OR any alert is not 'Normal'
alert_columns = ['HR_Alert', 'SpO2_Alert', 'BP_Alert', 'Temp_Alert']
df['Health_Issue_Detected'] = 0 # Initialize with 0 (No Issue)

# Conditions for detecting an issue
issue_conditions = (df['FallDetected'] == 'Yes') | \
                   (df['HR_Alert'] != 'Normal') | \
                   (df['SpO2_Alert'] != 'Normal') | \
                   (df['BP_Alert'] != 'Normal') | \
                   (df['Temp_Alert'] != 'Normal')

df.loc[issue_conditions, 'Health_Issue_Detected'] = 1

print("\nCreated 'Health_Issue_Detected' target variable.")
print("Value counts for 'Health_Issue_Detected':")
print(df['Health_Issue_Detected'].value_counts(normalize=True) * 100) # Show percentage

# Define Features (X) and Target (y)
# Use the primary sensor readings as features
# Do NOT use the individual alert columns or FallDetected as features,
# as they are used to *define* the target (avoid data leakage).
feature_columns = ['HeartRate', 'SpO2', 'SystolicBP', 'DiastolicBP', 'BodyTemp', 'DataAccuracy']
X = df[feature_columns]
y = df['Health_Issue_Detected']

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print("\nFeatures used:")
print(feature_columns)

# --- 3. Data Splitting ---
print("\n--- Splitting Data ---")
# Split into training and testing sets (e.g., 80% train, 20% test)
# Use stratify=y to ensure the proportion of classes is the same in train and test sets,
# which is important for potentially imbalanced datasets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print("Distribution in Training Set:\n", y_train.value_counts(normalize=True))
print("Distribution in Test Set:\n", y_test.value_counts(normalize=True))


# --- 4. Model Training (Random Forest) ---
print("\n--- Training Random Forest Model ---")

# Initialize the Random Forest Classifier
# We start with reasonably complex settings, reflecting the "large tree/more layers" idea
# class_weight='balanced' is CRUCIAL if the classes are imbalanced
rf_classifier = RandomForestClassifier(
    n_estimators=200,       # Increase number of trees for potentially better performance
    max_depth=None,         # Allow trees to grow deep (can be tuned later)
    min_samples_split=5,    # Minimum samples required to split an internal node
    min_samples_leaf=3,     # Minimum number of samples required to be at a leaf node
    random_state=42,        # For reproducibility
    class_weight='balanced', # Adjust weights inversely proportional to class frequencies
    n_jobs=-1               # Use all available CPU cores
)

start_time = time.time()
rf_classifier.fit(X_train, y_train)
end_time = time.time()
print(f"Model training completed in {end_time - start_time:.2f} seconds.")


# --- 5. Model Evaluation ---
print("\n--- Evaluating Model Performance ---")

# Predict on the Test Set
y_pred = rf_classifier.predict(X_test)
y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1] # Probabilities for the positive class

# Calculate Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f} (Of predicted issues, how many were real?)")
print(f"Recall:    {recall:.4f} (Of real issues, how many were detected?)")
print(f"F1-Score:  {f1:.4f} (Harmonic mean of precision and recall)")
print(f"ROC AUC:   {roc_auc:.4f} (Model's ability to distinguish between classes)")

print("\nConfusion Matrix:")
print(conf_matrix)
# Plotting the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Issue', 'Issue Detected'],
            yticklabels=['No Issue', 'Issue Detected'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Issue', 'Issue Detected']))

# --- 6. Feature Importance ---
print("\n--- Feature Importance ---")
importances = rf_classifier.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance from Random Forest')
plt.tight_layout()
plt.show()


# --- 7. Further Steps & Considerations (Optional) ---
print("\n--- Further Steps & Considerations ---")
print("1.  **Hyperparameter Tuning:** Use GridSearchCV or RandomizedSearchCV to find optimal parameters for the RandomForestClassifier (e.g., n_estimators, max_depth, min_samples_split, min_samples_leaf).")
print("2.  **Cross-Validation:** Implement k-fold cross-validation during training for a more robust estimate of model performance.")
print("3.  **Try Other Models:** Experiment with Gradient Boosting models like XGBoost, LightGBM, or CatBoost, which often provide top-tier performance on tabular data.")
print("4.  **Advanced Feature Engineering:** Create new features (e.g., pulse pressure = SystolicBP - DiastolicBP, ratios, moving averages if you have time-series data).")
print("5.  **Threshold Adjustment:** Depending on whether minimizing false alarms (higher precision) or minimizing missed detections (higher recall) is more critical, adjust the prediction threshold using predicted probabilities (y_pred_proba).")
print("6.  **Real-time Implementation:** This code trains and evaluates. For 24/7 monitoring, you'll need to deploy this model into a system that receives real-time sensor data, preprocesses it, and makes predictions.")
print("7.  **Data Quality:** The 'DataAccuracy' feature might be very important. Investigate its impact and how reliable the sensor data is in general.")