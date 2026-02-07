"""
Data Preprocessing Module

Handles data loading, cleaning, feature engineering, and transformation
for the churn analytics pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses telecom customer data for churn analysis.
    
    Responsibilities:
    - Load and validate data
    - Handle missing values
    - Engineer features
    - Encode categorical variables
    - Scale numerical features
    - Split train/test sets
    """
    
    def __init__(self, data_path, test_size=0.2, random_state=42):
        """
        Initialize preprocessor with data path.
        
        Args:
            data_path (str): Path to CSV file
            test_size (float): Proportion of test set 
            random_state (int): Random seed for reproducibility
        """
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def load_data(self):
        """
        Load data from CSV file.
        
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def clean_data(self, df):
        """
        Clean and preprocess data.
        
        Args:
            df (pd.DataFrame): Raw dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        df = df.copy()
        
        # Remove customer ID if present (not predictive)
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        # Handle TotalCharges (sometimes stored as string)
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            # Fill missing TotalCharges (new customers) with MonthlyCharges
            df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])
        
        # Convert target variable to binary (0/1)
        if 'Churn' in df.columns:
            df['Churn'] = df['Churn'].str.strip().str.lower().map({'yes': 1, 'no': 0})
        
        # Log cleaning summary
        missing_before = df.isnull().sum().sum()
        df = df.dropna()
        missing_after = df.isnull().sum().sum()
        
        logger.info(f"Data cleaning: {missing_before} missing values handled")
        logger.info(f"Final dataset: {df.shape[0]} rows")
        
        return df
    
    def engineer_features(self, df):
        """
        Create engineered features for better prediction.
        
        Args:
            df (pd.DataFrame): Cleaned dataframe
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        df = df.copy()
        
        # Tenure-based features
        if 'tenure' in df.columns:
            df['tenure_group'] = pd.cut(df['tenure'], 
                                       bins=[0, 12, 24, 48, 72],
                                       labels=['0-1 year', '1-2 years', '2-4 years', '4+ years'])
            df['is_new_customer'] = (df['tenure'] <= 6).astype(int)
        
        # Charges-based features
        if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
            df['avg_monthly_spend'] = df['TotalCharges'] / (df['tenure'].clip(lower=1))
            df['charge_to_tenure_ratio'] = df['MonthlyCharges'] / (df['tenure'].clip(lower=1))
        
        # Service usage features
        target_services = ['onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies']
        service_cols = [col for col in df.columns if 'service' in col.lower() or col.lower() in target_services]
        
        if service_cols:
            # Count number of services used
            df['num_services'] = (df[service_cols]=='Yes').sum(axis=1)
        
        # Contract type risk indicator
        if 'Contract' in df.columns:
            df['high_risk_contract'] = (df['Contract'] == 'Month-to-month').astype(int)
        
        logger.info(f"Feature engineering complete: {len(df.columns)} total features")
        
        return df
    
    def encode_features(self, df):
        """
        Encode categorical variables.
        
        Args:
            df (pd.DataFrame): Dataframe with raw features
            
        Returns:
            pd.DataFrame: Dataframe with encoded features
        """
        df = df.copy()
        
        # Identify categorical columns (excluding target)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col != 'Churn']
        
        # Binary encoding for Yes/No columns
        binary_cols = []
        for col in categorical_cols:
            if set(df[col].str.lower().unique()).issubset({'yes', 'no', 'no internet service', 'no phone service'}):
                df[col] = df[col].map({
                    'Yes': 1, 
                    'No': 0, 
                    'No internet service': 0,
                    'No phone service': 0
                })
                binary_cols.append(col)
        
        # One-hot encoding for multi-category columns
        remaining_categorical = [col for col in categorical_cols if col not in binary_cols]
        
        if remaining_categorical:
            df = pd.get_dummies(df, columns=remaining_categorical, drop_first=True)
        
        logger.info(f"Encoding complete: {len(binary_cols)} binary, "
                   f"{len(remaining_categorical)} one-hot encoded")
        
        return df
    
    def prepare_data(self):
        """
        Execute full preprocessing pipeline.
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test, processed_df)
        """
        # Load and clean
        df = self.load_data()
        df = self.clean_data(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Encode features
        df = self.encode_features(df)
        
        # Separate features and target
        # Identify the actual column name used in the data (e.g., 'churn', 'CHURN', or 'Churn')
        actual_churn_col = next((col for col in df.columns if col.lower() == 'churn'), None)
        # Safety check using the dynamic name
        if actual_churn_col is None:
            raise ValueError("Target variable 'Churn' not found in dataset")

        # Separate features and target using the confirmed name
        X = df.drop(actual_churn_col, axis=1)
        y = df[actual_churn_col]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale numerical features
        X_train = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X.columns,
            index=X_train.index
        )
        
        X_test = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X.columns,
            index=X_test.index
        )
        
        logger.info("Data preparation complete")
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, df
    
    def transform(self, df):
        """
        Transform new data using fitted preprocessor.
        
        Args:
            df (pd.DataFrame): New data to transform
            
        Returns:
            np.ndarray: Transformed feature array
        """
        df = df.copy()
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            for feat in missing_features:
                df[feat] = 0
        
        # Ensure correct column order
        df = df[self.feature_names]
        
        # Scale
        return self.scaler.transform(df)