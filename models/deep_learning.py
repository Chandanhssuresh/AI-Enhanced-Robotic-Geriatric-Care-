import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, multilabel_confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns # For plotting confusion matrices

print(f"TensorFlow Version: {tf.__version__}")

# --- Configuration ---
DATA_FILE = '/home/chandansuresh/perfect_dataset.csv'
TEST_SIZE = 0.2  # 20% of data for testing
RANDOM_STATE = 42 # For reproducibility
EPOCHS = 50        # Max number of training epochs
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10 # Stop training if no improvement after 10 epochs
PREDICTION_THRESHOLD = 0.5 # Threshold for converting probabilities to binary predictions

# --- 1. Load Data ---
print(f"\n--- Loading Data from {DATA_FILE} ---")
try:
    df = pd.read_csv(DATA_FILE)
    print(f"Successfully loaded {len(df)} rows.")
    print("Dataset head:")
    print(df.head())
    print("\nDataset info:")
    df.info()
    print("\nValue counts for alert columns:")
    for col in ['Heart Rate Alert', 'SpO2 Level Alert', 'Blood Pressure Alert', 'Temperature Alert']:
        print(f"\n{col}:")
        print(df[col].value_counts())
except FileNotFoundError:
    print(f"Error: File not found at '{DATA_FILE}'. Please ensure the file exists.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- 2. Preprocess Data ---
print("\n--- Preprocessing Data ---")

# Define features (X) and original targets
feature_cols = [
    'Heart Rate (bpm)', 'SpO2 Level (%)', 'Systolic Blood Pressure (mmHg)',
    'Diastolic Blood Pressure (mmHg)', 'Body Temperature (°C)',
    'Fall Detection', 'Data Accuracy (%)'
]
original_target_cols = [
    'Heart Rate Alert', 'SpO2 Level Alert', 'Blood Pressure Alert', 'Temperature Alert'
]

# Check for missing values (optional, but good practice)
print(f"\nMissing values per column:\n{df[feature_cols + original_target_cols].isnull().sum()}")
# If missing values exist: df.dropna(inplace=True) or df.fillna(...)

# Handle 'Fall Detection' (Categorical Input Feature)
# Ensure the column exists before mapping
if 'Fall Detection' in df.columns:
    df['Fall Detection_encoded'] = df['Fall Detection'].map({'Yes': 1, 'No': 0})
    # Add the encoded column to features and remove the original if desired
    feature_cols.remove('Fall Detection')
    feature_cols.append('Fall Detection_encoded')
    print("'Fall Detection' mapped to 0/1.")
else:
    print("Warning: 'Fall Detection' column not found in the dataset.")


# --- Define Targets (Multi-label Binary Classification) ---
# Create binary targets: 1 if alert is NOT 'Normal', 0 if it IS 'Normal'
binary_target_cols = []
for col in original_target_cols:
    if col in df.columns:
        new_col_name = f'{col}_ALERT' # Changed suffix for clarity
        df[new_col_name] = df[col].apply(lambda x: 0 if pd.isna(x) or x.strip().lower() == 'normal' else 1)
        binary_target_cols.append(new_col_name)
        print(f"Created binary target: {new_col_name} (1 = Alert, 0 = Normal)")
    else:
         print(f"Warning: Target column '{col}' not found.")

if not binary_target_cols:
    print("Error: No target columns could be created. Please check column names.")
    exit()

print("\nValue counts for binary targets:")
for col in binary_target_cols:
     print(df[col].value_counts(dropna=False)) # Show NaN counts too if any

# Separate features (X) and targets (y)
try:
    X = df[feature_cols].astype(float) # Ensure numeric types
    y = df[binary_target_cols].astype(int) # Targets are 0/1 integers
except KeyError as e:
    print(f"Error: Column missing for X or y creation: {e}")
    print(f"Available columns: {df.columns.tolist()}")
    exit()
except ValueError as e:
    print(f"Error converting columns to numeric: {e}. Check data for non-numeric values.")
    # Consider df[col].replace({'some_string': np.nan}).astype(float)
    exit()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y if y.shape[1] > 0 else None # Stratify if possible for multi-label (might be complex)
)
print(f"\nData split: {X_train.shape[0]} train, {X_test.shape[0]} test samples.")
print(f"Feature shape: {X_train.shape[1]}, Target shape: {y_train.shape[1]}")

# Scale numerical features (Fit on train, transform train and test)
# Identify numerical columns *within X* (excluding already encoded binary features like 'Fall Detection_encoded')
numerical_features_to_scale = [col for col in feature_cols if col != 'Fall Detection_encoded']

scaler = StandardScaler()
# Fit only on training data
X_train[numerical_features_to_scale] = scaler.fit_transform(X_train[numerical_features_to_scale])
# Transform both train and test data
X_test[numerical_features_to_scale] = scaler.transform(X_test[numerical_features_to_scale])

print("Numerical features scaled using StandardScaler.")
print("\nSample of preprocessed X_train head:")
print(X_train.head())

# --- 3. Build the Deep Learning Model (Multi-label Classification) ---
print("\n--- Building Keras Model ---")

input_dim = X_train.shape[1]
output_dim = y_train.shape[1] # Number of binary target labels

def build_model(input_dim, output_dim):
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,), name="Input_Layer"),
            layers.Dense(128, activation="relu", name="Dense_1"),
            layers.BatchNormalization(), # Helps stabilize training
            layers.Dropout(0.4),         # Increased dropout
            layers.Dense(64, activation="relu", name="Dense_2"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu", name="Dense_3"), # Added another layer
            layers.BatchNormalization(),
            layers.Dense(output_dim, activation="sigmoid", name="Output_Layer") # Sigmoid for multi-label binary
        ],
        name="Health_Prediction_Model"
    )
    return model

model = build_model(input_dim, output_dim)
model.summary()

# --- 4. Compile the Model ---
print("\n--- Compiling Model ---")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',    # Correct loss for multi-label binary classification
    metrics=[
        'binary_accuracy',         # Accuracy for each label independently
        tf.keras.metrics.AUC(name='auc'), # Area under ROC curve
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)
print("Model compiled successfully.")

# --- 5. Train the Model ---
print("\n--- Training Model ---")

# Add EarlyStopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',              # Monitor validation loss
    patience=EARLY_STOPPING_PATIENCE,
    verbose=1,
    restore_best_weights=True,       # Restore weights from the epoch with the best val_loss
    mode='min'                       # We want to minimize loss
)

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],      # Add the callback here
    verbose=1                        # Show progress bar per epoch
)

print("Training finished.")

# --- Plot Training History ---
def plot_history(history):
    print("\n--- Plotting Training History ---")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Training History')

    # Plot Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True)


    # Plot Binary Accuracy
    axes[0, 1].plot(history.history['binary_accuracy'], label='Training Binary Accuracy')
    axes[0, 1].plot(history.history['val_binary_accuracy'], label='Validation Binary Accuracy')
    axes[0, 1].set_title('Binary Accuracy')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot AUC
    axes[1, 0].plot(history.history['auc'], label='Training AUC')
    axes[1, 0].plot(history.history['val_auc'], label='Validation AUC')
    axes[1, 0].set_title('AUC (Area Under Curve)')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot Precision & Recall (Example - might be noisy per epoch)
    axes[1, 1].plot(history.history['precision'], label='Training Precision')
    axes[1, 1].plot(history.history['val_precision'], label='Validation Precision')
    axes[1, 1].plot(history.history['recall'], label='Training Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title('Precision & Recall')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(True)


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()

plot_history(history)

# --- 6. Evaluate the Model ---
print("\n--- Evaluating Model on Test Data ---")
evaluation_results = model.evaluate(X_test, y_test, verbose=0)

print("\nOverall Test Metrics:")
for name, value in zip(model.metrics_names, evaluation_results):
    print(f"{name}: {value:.4f}")

# Get predictions (probabilities)
y_pred_proba = model.predict(X_test)

# Convert probabilities to binary predictions based on the threshold
y_pred_binary = (y_pred_proba > PREDICTION_THRESHOLD).astype(int)

print(f"\n--- Classification Report (Threshold = {PREDICTION_THRESHOLD}) ---")
# Note: classification_report provides precision, recall, f1-score per label
# 'support' is the number of true instances for each label in y_test
print(classification_report(y_test, y_pred_binary, target_names=binary_target_cols, zero_division=0))

# Overall accuracy (Exact Match Ratio) - Proportion of samples where all labels are predicted correctly
exact_match_accuracy = accuracy_score(y_test, y_pred_binary)
print(f"\nExact Match Accuracy (all labels correct): {exact_match_accuracy:.4f}")
print("(This is a strict metric, often low in multi-label tasks)")

# --- Multi-label Confusion Matrices ---
print("\n--- Multi-label Confusion Matrices (per label) ---")
mcm = multilabel_confusion_matrix(y_test, y_pred_binary)

plt.figure(figsize=(12, output_dim * 3)) # Adjust figure size based on number of labels
for i, label_name in enumerate(binary_target_cols):
    plt.subplot(output_dim, 2, i*2 + 1) # Adjust subplot grid if many labels
    sns.heatmap(mcm[i], annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Normal', 'Predicted ALERT'],
                yticklabels=['Actual Normal', 'Actual ALERT'])
    plt.title(f'Confusion Matrix: {label_name}')
    tn, fp, fn, tp = mcm[i].ravel()
    print(f"\nConfusion Matrix for {label_name}:")
    print(f"  TN: {tn} (Correct Normal)")
    print(f"  FP: {fp} (Incorrect Alert - Type I Error)")
    print(f"  FN: {fn} (Missed Alert - Type II Error)")
    print(f"  TP: {tp} (Correct Alert)")
    print("-" * 20)
plt.tight_layout()
plt.show()


# --- 7. Making Predictions on New Data ---
print("\n--- Example Prediction on New Data ---")

# Create sample new data points (as a dictionary or DataFrame)
# IMPORTANT: Structure must match the training features EXACTLY before scaling
# Use original categorical values ('Yes'/'No') for features that were mapped
new_data_samples = [
    { # Example 1: Potentially concerning readings
        'Heart Rate (bpm)': 115, 'SpO2 Level (%)': 89,
        'Systolic Blood Pressure (mmHg)': 150, 'Diastolic Blood Pressure (mmHg)': 95,
        'Body Temperature (°C)': 38.8, 'Fall Detection': 'No', 'Data Accuracy (%)': 94
    },
    { # Example 2: Normal readings
        'Heart Rate (bpm)': 70, 'SpO2 Level (%)': 98,
        'Systolic Blood Pressure (mmHg)': 115, 'Diastolic Blood Pressure (mmHg)': 75,
        'Body Temperature (°C)': 36.7, 'Fall Detection': 'No', 'Data Accuracy (%)': 99
    },
     { # Example 3: Fall detected, slightly low SpO2
        'Heart Rate (bpm)': 85, 'SpO2 Level (%)': 92,
        'Systolic Blood Pressure (mmHg)': 125, 'Diastolic Blood Pressure (mmHg)': 82,
        'Body Temperature (°C)': 37.1, 'Fall Detection': 'Yes', 'Data Accuracy (%)': 90
    }
]

new_data_df = pd.DataFrame(new_data_samples)
print("New data (raw):")
print(new_data_df)

# Preprocess the new data *using the same steps and fitted objects*
# 1. Encode 'Fall Detection'
if 'Fall Detection_encoded' in feature_cols: # Check if the encoded column is expected
    new_data_df['Fall Detection_encoded'] = new_data_df['Fall Detection'].map({'Yes': 1, 'No': 0})
    new_data_processed = new_data_df.drop(columns=['Fall Detection']) # Drop original if encoded
else:
    new_data_processed = new_data_df.copy()

# 2. Ensure correct column order and types (matching X_train columns)
# Reindex to guarantee column order matches the model's input
try:
    new_data_processed = new_data_processed[feature_cols].astype(float)
except KeyError as e:
    print(f"Error: Column mismatch during new data preprocessing: {e}")
    print(f"Expected columns: {feature_cols}")
    print(f"Columns in new data: {new_data_processed.columns.tolist()}")
    exit()


# 3. Scale numerical features using the *fitted* scaler
new_data_processed[numerical_features_to_scale] = scaler.transform(new_data_processed[numerical_features_to_scale])

print("\nNew data (preprocessed):")
print(new_data_processed)

# Make prediction
prediction_proba_new = model.predict(new_data_processed)
prediction_binary_new = (prediction_proba_new > PREDICTION_THRESHOLD).astype(int)

print("\n--- Predictions for New Data ---")
for i in range(len(new_data_df)):
    print(f"\nSample {i+1}:")
    print(f" Raw Input: {new_data_samples[i]}")
    print(f" Prediction Probabilities: {prediction_proba_new[i]}")
    print(f" Binary Predictions (Threshold > {PREDICTION_THRESHOLD}): {prediction_binary_new[i]}")
    print(" Interpreted Alerts:")
    for j, label_name in enumerate(binary_target_cols):
        alert_status = "ALERT" if prediction_binary_new[i, j] == 1 else "Normal"
        original_label_name = label_name.replace('_ALERT','')
        print(f"  - {original_label_name}: {alert_status} (Prob: {prediction_proba_new[i, j]:.3f})")

# --- Optional: Save the Model and Scaler ---
# print("\n--- Saving Model and Scaler ---")
# model.save('health_prediction_model.keras')
# import joblib
# joblib.dump(scaler, 'health_data_scaler.joblib')
# print("Model saved as 'health_prediction_model.keras'")
# print("Scaler saved as 'health_data_scaler.joblib'")

# To load later:
# from tensorflow import keras
# import joblib
# loaded_model = keras.models.load_model('health_prediction_model.keras')
# loaded_scaler = joblib.load('health_data_scaler.joblib')
# print("Model and scaler loaded.")