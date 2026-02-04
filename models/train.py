"""
ML Training Pipeline
Trains, evaluates, and saves the Logistic Regression model
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPipeline:
    """Handles model training, evaluation, and persistence."""
    
    def __init__(self, model_dir: str = "models"):
        """Initialize the pipeline."""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metrics = None
        self.feature_importance = None
        
    def train(self, X: pd.DataFrame, y: pd.Series, 
              test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Train the model with proper scaling.
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of test set
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with training results
        """
        logger.info("Starting model training...")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = LogisticRegression(
            max_iter=1000, random_state=random_state, 
            class_weight='balanced', solver='lbfgs'
        )
        self.model.fit(X_train_scaled, y_train)
        logger.info("Model training completed")
        
        # Evaluate
        results = self._evaluate(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Calculate feature importance
        self._calculate_feature_importance()
        
        return results
    
    def _evaluate(self, X_train_scaled, X_test_scaled, y_train, y_test) -> Dict[str, Any]:
        """Evaluate model performance."""
        logger.info("Evaluating model...")
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'auc_roc': float(roc_auc_score(y_test, y_pred_proba)),
            'train_accuracy': float(accuracy_score(y_train, self.model.predict(X_train_scaled))),
            'test_accuracy': float(accuracy_score(y_test, y_pred)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        self.metrics = metrics
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, "
                   f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
        
        return metrics
    
    def _calculate_feature_importance(self):
        """Calculate feature importance from model coefficients."""
        if self.model is None or self.feature_names is None:
            return
        
        # Normalize coefficients for importance
        coefficients = np.abs(self.model.coef_[0])
        normalized = coefficients / coefficients.sum()
        
        self.feature_importance = {
            name: float(importance)
            for name, importance in zip(self.feature_names, normalized)
        }
        
        # Sort by importance
        self.feature_importance = dict(
            sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
    
    def save_model(self, model_name: str = "placement_model") -> Tuple[str, str]:
        """Save model and scaler to files."""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Call train() first.")
        
        model_path = self.model_dir / f"{model_name}.pkl"
        scaler_path = self.model_dir / f"{model_name}_scaler.pkl"
        metrics_path = self.model_dir / f"{model_name}_metrics.json"
        importance_path = self.model_dir / f"{model_name}_importance.json"
        
        # Save model and scaler
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save metrics and importance
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        with open(importance_path, 'w') as f:
            json.dump(self.feature_importance, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
        logger.info(f"Metrics saved to {metrics_path}")
        logger.info(f"Importance saved to {importance_path}")
        
        return str(model_path), str(scaler_path)
    
    def load_model(self, model_path: str, scaler_path: str):
        """Load model and scaler from files."""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        logger.info(f"Model loaded from {model_path}")
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get model metrics."""
        return self.metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        return self.feature_importance

