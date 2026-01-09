import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import FunctionTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Flatten, 
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ReduceLROnPlateau, 
    ModelCheckpoint
)
from tensorflow.keras.regularizers import l2
from scipy.stats import boxcox

class AdvancedHealthPredictor:
    def __init__(self, file_path):
        """
        Initialize the health prediction model
        
        Args:
        file_path (str): Path to the health dataset
        """
        self.file_path = file_path
        self.model = None
        self.preprocessor = None
        self.feature_selector = None
    
    def _boxcox_transform(self, X):
        """
        Apply Box-Cox transformation to handle skewed distributions
        
        Args:
        X (numpy.ndarray): Input features
        
        Returns:
        Transformed features
        """
        X_transformed = np.zeros_like(X, dtype=float)
        for i in range(X.shape[1]):
            # Add small constant to avoid issues with zero or negative values
            X_column = X[:, i] + 1e-10
            
            # Apply Box-Cox transformation
            try:
                X_transformed[:, i], _ = boxcox(X_column)
            except Exception:
                # Fallback to original values if transformation fails
                X_transformed[:, i] = X[:, i]
        
        return X_transformed
    
    def advanced_preprocessing(self, X, y=None):
        """
        Advanced preprocessing with multiple techniques
        
        Args:
        X (pandas.DataFrame): Input features
        y (pandas.Series, optional): Target variable
        
        Returns:
        Preprocessed features
        """
        # Identify column types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        
        # Preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                # Numeric features preprocessing
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('boxcox', FunctionTransformer(self._boxcox_transform)),
                    ('scaler', RobustScaler())
                ]), numeric_features),
                
                # Categorical features preprocessing
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_features)
            ])
        
        # Fit and transform
        X_transformed = preprocessor.fit_transform(X)
        
        # Feature selection
        if y is not None:
            selector = SelectKBest(f_classif, k=min(20, X_transformed.shape[1]))
            X_selected = selector.fit_transform(X_transformed, y)
            self.feature_selector = selector
        else:
            X_selected = X_transformed
        
        self.preprocessor = preprocessor
        return X_selected
    
    def create_cnn_model(self, input_shape):
        """
        Create a 1D Convolutional Neural Network model
        
        Args:
        input_shape (tuple): Shape of input features
        
        Returns:
        Compiled Keras model
        """
        model = Sequential([
            # Convolutional layers
            Conv1D(64, kernel_size=3, activation='relu', 
                   input_shape=input_shape, 
                   kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            Conv1D(128, kernel_size=3, activation='relu', 
                   kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.4),
            
            # Flatten and fully connected layers
            Flatten(),
            
            Dense(256, activation='relu', 
                  kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(128, activation='relu', 
                  kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.4),
            
            # Output layer
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                     tf.keras.metrics.Precision(), 
                     tf.keras.metrics.Recall()]
        )
        
        return model
    
    def train_model(self, test_size=0.2, random_state=42):
        """
        Train the CNN model
        
        Args:
        test_size (float): Proportion of test data
        random_state (int): Random seed for reproducibility
        
        Returns:
        Training history
        """
        # Load data
        df = pd.read_csv(self.file_path)
        
        # Separate features and target
        # Assuming last column is binary target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Advanced preprocessing
        X_processed = self.advanced_preprocessing(X, y)
        
        # Reshape for CNN input (samples, timesteps, features)
        X_processed = X_processed.reshape(
            X_processed.shape[0], 
            X_processed.shape[1], 
            1
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )
        
        # Create model
        self.model = self.create_cnn_model(
            input_shape=(X_processed.shape[1], 1)
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_accuracy', 
            patience=15, 
            restore_best_weights=True
        )
        
        lr_reducer = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=8, 
            min_lr=0.00001
        )
        
        model_checkpoint = ModelCheckpoint(
            'best_health_model.h5', 
            monitor='val_accuracy', 
            save_best_only=True
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=64,
            validation_split=0.2,
            callbacks=[early_stopping, lr_reducer, model_checkpoint],
            verbose=1
        )
        
        # Evaluate model
        test_loss, test_accuracy, precision, recall = self.model.evaluate(
            X_test, y_test, verbose=0
        )
        
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        
        return history
    
    def predict(self, new_data_path):
        """
        Make predictions on new data
        
        Args:
        new_data_path (str): Path to new data CSV
        
        Returns:
        Predictions
        """
        # Load new data
        new_df = pd.read_csv(new_data_path)
        
        # Preprocess new data
        X_new = new_df.iloc[:, :-1]
        X_processed = self.advanced_preprocessing(X_new)
        
        # Reshape for CNN
        X_processed = X_processed.reshape(
            X_processed.shape[0], 
            X_processed.shape[1], 
            1
        )
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        
        return predictions

# Example usage
if __name__ == "__main__":
    # Path to your health dataset
    dataset_path = '/home/chandansuresh/perfect_dataset.csv'
    
    # Initialize and train the model
    health_predictor = AdvancedHealthPredictor(dataset_path)
    training_history = health_predictor.train_model()
