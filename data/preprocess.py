"""
Data Preprocessing Module
Handles data loading, cleaning, and preparation for ML pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles all data preprocessing tasks."""
    
    def __init__(self, data_path: str):
        """Initialize preprocessor with data path."""
        self.data_path = Path(data_path)
        self.df = None
        self.feature_columns = [
            'cgpa', 'internships', 'projects', 'communication_skills',
            'problem_solving', 'technical_skills', 'leadership',
            'teamwork', 'adaptability'
        ]
        self.target_column = 'placed'
        
    def load_data(self) -> pd.DataFrame:
        """Load CSV data."""
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def validate_data(self) -> bool:
        """Validate data integrity."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Check for missing values
        missing = self.df.isnull().sum()
        if missing.any():
            logger.warning(f"Missing values found:\n{missing}")
            return False
        
        # Check for duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows")
        
        # Check data types
        for col in self.feature_columns + [self.target_column]:
            if col not in self.df.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        logger.info("Data validation passed")
        return True
    
    def get_features_and_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract features and target variable."""
        if self.df is None:
            self.load_data()
        
        X = self.df[self.feature_columns].copy()
        y = self.df[self.target_column].copy()
        
        return X, y
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the data."""
        if self.df is None:
            self.load_data()
        
        return {
            'total_records': len(self.df),
            'features': self.feature_columns,
            'target': self.target_column,
            'placed_count': (self.df[self.target_column] == 1).sum(),
            'not_placed_count': (self.df[self.target_column] == 0).sum(),
            'placement_rate': (self.df[self.target_column] == 1).sum() / len(self.df) * 100,
            'feature_ranges': {
                col: {
                    'min': self.df[col].min(),
                    'max': self.df[col].max(),
                    'mean': self.df[col].mean()
                }
                for col in self.feature_columns
            }
        }


def load_and_preprocess(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Convenience function to load and preprocess data."""
    preprocessor = DataPreprocessor(data_path)
    preprocessor.load_data()
    preprocessor.validate_data()
    return preprocessor.get_features_and_target()

