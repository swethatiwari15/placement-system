"""
Helper Functions and Utilities
Common utilities for prediction and data processing
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from utils.config import (
    MODEL_FILE, SCALER_FILE, METRICS_FILE, IMPORTANCE_FILE,
    FEATURE_NAMES, FEATURE_RANGES, FEATURE_LABELS, PLACEMENT_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD, LOW_CONFIDENCE_THRESHOLD
)
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model loading and prediction."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.model = None
        self.scaler = None
        self.metrics = None
        self.feature_importance = None
        self._load_model()
        self._initialized = True
    
    def _load_model(self):
        """Load model and scaler."""
        try:
            if MODEL_FILE.exists() and SCALER_FILE.exists():
                with open(MODEL_FILE, 'rb') as f:
                    self.model = pickle.load(f)
                with open(SCALER_FILE, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("Model loaded successfully")
                
                # Load metrics if available
                if METRICS_FILE.exists():
                    with open(METRICS_FILE, 'r') as f:
                        self.metrics = json.load(f)
                
                # Load feature importance if available
                if IMPORTANCE_FILE.exists():
                    with open(IMPORTANCE_FILE, 'r') as f:
                        self.feature_importance = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
    
    def is_available(self) -> bool:
        """Check if model is available."""
        return self.model is not None and self.scaler is not None
    
    def predict(self, features_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Make prediction for a single student.
        
        Args:
            features_dict: Dictionary with feature values
            
        Returns:
            Prediction result with probability and confidence
        """
        if not self.is_available():
            raise RuntimeError("Model is not available")
        
        # Create DataFrame
        X = pd.DataFrame([features_dict])
        
        # Ensure correct order
        X = X[FEATURE_NAMES]
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Calculate confidence
        prob_placed = float(probabilities[1])
        confidence = max(probabilities)
        confidence_level = self._get_confidence_level(confidence)
        
        return {
            'prediction': int(prediction),
            'placed': bool(prediction == 1),
            'probability': prob_placed,
            'confidence': float(confidence),
            'confidence_level': confidence_level
        }
    
    @staticmethod
    def _get_confidence_level(confidence: float) -> str:
        """Get confidence level label."""
        if confidence >= HIGH_CONFIDENCE_THRESHOLD:
            return "High"
        elif confidence >= LOW_CONFIDENCE_THRESHOLD:
            return "Medium"
        else:
            return "Low"


def validate_student_input(features_dict: Dict[str, float]) -> Tuple[bool, str]:
    """
    Validate student input data.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    for feature, value in features_dict.items():
        if feature not in FEATURE_RANGES:
            return False, f"Unknown feature: {feature}"
        
        min_val, max_val = FEATURE_RANGES[feature]
        
        if not isinstance(value, (int, float)):
            return False, f"{FEATURE_LABELS.get(feature, feature)} must be a number"
        
        if value < min_val or value > max_val:
            return False, f"{FEATURE_LABELS.get(feature, feature)} must be between {min_val} and {max_val}"
    
    return True, ""


def get_placement_insights(features_dict: Dict[str, float], prediction_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate human-readable insights for placement prediction.
    
    Returns:
        Dictionary with insights and recommendations
    """
    manager = ModelManager()
    
    insights = {
        'prediction': "✓ Likely to be Placed" if prediction_result['placed'] else "✗ May Face Challenges",
        'confidence': prediction_result['confidence_level'],
        'probability': f"{prediction_result['probability'] * 100:.1f}%",
        'strong_factors': [],
        'weak_factors': [],
        'recommendations': []
    }
    
    # Analyze features based on importance
    if manager.feature_importance:
        for feature, importance in sorted(manager.feature_importance.items(), 
                                         key=lambda x: x[1], reverse=True):
            if feature in features_dict:
                value = features_dict[feature]
                min_val, max_val = FEATURE_RANGES[feature]
                percentile = (value - min_val) / (max_val - min_val) * 100
                
                # Strong factors (above 70%)
                if percentile >= 70:
                    insights['strong_factors'].append({
                        'factor': FEATURE_LABELS.get(feature, feature),
                        'value': value,
                        'rating': 'Excellent' if percentile >= 85 else 'Good'
                    })
                # Weak factors (below 40%)
                elif percentile < 40:
                    insights['weak_factors'].append({
                        'factor': FEATURE_LABELS.get(feature, feature),
                        'value': value,
                        'rating': 'Needs Improvement'
                    })
    
    # Generate recommendations
    insights['recommendations'] = generate_recommendations(features_dict, prediction_result['placed'])
    
    return insights


def generate_recommendations(features_dict: Dict[str, float], is_placed: bool) -> List[str]:
    """Generate actionable recommendations."""
    recommendations = []
    
    if not is_placed:
        # Suggest improvements for non-placed students
        min_val, max_val = FEATURE_RANGES['cgpa']
        if features_dict.get('cgpa', 0) < 7.0:
            recommendations.append("Focus on improving CGPA to above 7.0 for better placement chances")
        
        if features_dict.get('internships', 0) == 0:
            recommendations.append("Take at least 1-2 internships to gain practical experience")
        
        if features_dict.get('projects', 0) < 2:
            recommendations.append("Complete 2-3 project-based courses or independent projects")
        
        if features_dict.get('technical_skills', 1) < 6:
            recommendations.append("Enhance technical skills through online courses or certifications")
        
        if features_dict.get('communication_skills', 1) < 6:
            recommendations.append("Improve communication and soft skills through workshops or groups")
    else:
        # Suggestions for placed students
        recommendations.append("Congratulations! Maintain your strong performance for career growth")
        recommendations.append("Continue developing your skills to stand out in your role")
    
    return recommendations


def get_model_metrics() -> Dict[str, Any]:
    """Get model performance metrics."""
    manager = ModelManager()
    return manager.metrics or {}


def get_feature_importance() -> Dict[str, float]:
    """Get feature importance ranking."""
    manager = ModelManager()
    return manager.feature_importance or {}

