"""
Churn Prediction Model

Implements logistic regression for customer churn prediction with
comprehensive evaluation metrics.
"""

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import logging

logger = logging.getLogger(__name__)


class ChurnPredictor:
    
    def __init__(self, threshold=0.5, random_state=42):
        """
        Initialize churn predictor.
        
        Args:
            threshold (float): Classification threshold for predictions
            random_state (int): Random seed for reproducibility
        """
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            solver='lbfgs',
            class_weight='balanced'  # Handles class imbalance
        )
        self.threshold = threshold
        self.is_fitted = False
        
    def train(self, X_train, y_train):
        """
        Train the logistic regression model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            self: Fitted model
        """
        logger.info("Training logistic regression model...")
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Log training performance
        train_accuracy = self.model.score(X_train, y_train)
        logger.info(f"Training accuracy: {train_accuracy:.1%}")
        
        return self
    
    def predict(self, X):
        """
        Predict churn (binary outcome).
        
        Args:
            X: Feature matrix
            
        Returns:
            np.ndarray: Binary predictions (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        probabilities = self.predict_proba(X)
        return (probabilities >= self.threshold).astype(int)
    
    def predict_proba(self, X):
        """
        Predict churn probability.
        
        Args:
            X: Feature matrix
            
        Returns:
            np.ndarray: Probability scores (0 to 1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        # Return probability of positive class (churn)
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X_test, y_test, verbose=True):
        """
        Comprehensive model evaluation.
        
        Args:
            X_test: Test features
            y_test: True test labels
            verbose (bool): Print detailed metrics
            
        Returns:
            dict: Dictionary of performance metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")
        
        # Predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        if verbose:
            logger.info("\nMODEL PERFORMANCE METRICS")
            logger.info("-" * 50)
            logger.info(f"Accuracy:  {metrics['accuracy']:.1%}")
            logger.info(f"Precision: {metrics['precision']:.1%}")
            logger.info(f"Recall:    {metrics['recall']:.1%}")
            logger.info(f"F1-Score:  {metrics['f1_score']:.1%}")
            logger.info(f"ROC-AUC:   {metrics['roc_auc']:.3f}")
            logger.info("-" * 50)
            
            # Business interpretation
            logger.info("\nBUSINESS INTERPRETATION:")
            logger.info(f"• Model correctly identifies {metrics['recall']:.0%} of customers who will churn")
            logger.info(f"• {metrics['precision']:.0%} of churn predictions are accurate")
            logger.info(f"• Overall prediction accuracy: {metrics['accuracy']:.0%}")
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            logger.info("\nCONFUSION MATRIX:")
            logger.info(f"True Negatives:  {cm[0,0]:,}")
            logger.info(f"False Positives: {cm[0,1]:,}")
            logger.info(f"False Negatives: {cm[1,0]:,}")
            logger.info(f"True Positives:  {cm[1,1]:,}")
        
        return metrics
    
    def get_feature_importance(self, feature_names):
        """
        Extract feature importance from model coefficients.
        
        Args:
            feature_names (list): List of feature names
            
        Returns:
            pd.DataFrame: Sorted feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        
        import pandas as pd
        
        # Get coefficients (log-odds)
        coefficients = self.model.coef_[0]
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        })
        
        # Sort by absolute value
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath):
        """
        Save trained model to disk.
        
        Args:
            filepath (str): Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save untrained model")
        
        joblib.dump(self, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """
        Load trained model from disk.
        
        Args:
            filepath (str): Path to saved model
            
        Returns:
            ChurnPredictor: Loaded model
        """
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def get_risk_segments(self, X, percentiles=[25, 50, 75]):
        """
        Segment customers by churn risk level.
        
        Args:
            X: Feature matrix
            percentiles (list): Percentile thresholds for segmentation
            
        Returns:
            np.ndarray: Risk segment labels
        """
        probabilities = self.predict_proba(X)
        
        segments = np.zeros(len(probabilities), dtype=object)
        
        thresholds = np.percentile(probabilities, percentiles)
        
        segments[probabilities <= thresholds[0]] = 'Low Risk'
        segments[(probabilities > thresholds[0]) & (probabilities <= thresholds[1])] = 'Medium Risk'
        segments[(probabilities > thresholds[1]) & (probabilities <= thresholds[2])] = 'High Risk'
        segments[probabilities > thresholds[2]] = 'Critical Risk'
        
        return segments
