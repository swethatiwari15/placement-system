"""
Configuration and Constants
Centralized configuration for the application
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
UTILS_DIR = PROJECT_ROOT / "utils"

# Model files
MODEL_FILE = MODELS_DIR / "placement_model.pkl"
SCALER_FILE = MODELS_DIR / "placement_model_scaler.pkl"
METRICS_FILE = MODELS_DIR / "placement_model_metrics.json"
IMPORTANCE_FILE = MODELS_DIR / "placement_model_importance.json"

# Data files
DATA_FILE = DATA_DIR / "placement_data.csv"

# Feature configuration
FEATURE_NAMES = [
    'cgpa',
    'internships',
    'projects',
    'communication_skills',
    'problem_solving',
    'technical_skills',
    'leadership',
    'teamwork',
    'adaptability'
]

FEATURE_RANGES = {
    'cgpa': (5.0, 10.0),
    'internships': (0, 5),
    'projects': (0, 10),
    'communication_skills': (1, 10),
    'problem_solving': (1, 10),
    'technical_skills': (1, 10),
    'leadership': (1, 10),
    'teamwork': (1, 10),
    'adaptability': (1, 10)
}

FEATURE_LABELS = {
    'cgpa': 'CGPA',
    'internships': 'Number of Internships',
    'projects': 'Number of Projects',
    'communication_skills': 'Communication Skills',
    'problem_solving': 'Problem Solving',
    'technical_skills': 'Technical Skills',
    'leadership': 'Leadership',
    'teamwork': 'Teamwork',
    'adaptability': 'Adaptability'
}

# Feature groups for UI organization
FEATURE_GROUPS = {
    'academics': {
        'label': 'Academic Profile',
        'features': ['cgpa']
    },
    'experience': {
        'label': 'Experience & Projects',
        'features': ['internships', 'projects']
    },
    'skills': {
        'label': 'Skills & Competencies',
        'features': ['communication_skills', 'problem_solving', 'technical_skills',
                    'leadership', 'teamwork', 'adaptability']
    }
}

# UI Configuration
THEME_COLOR = "#1f77b4"
ACCENT_COLOR = "#ff7f0e"
SUCCESS_COLOR = "#2ecc71"
DANGER_COLOR = "#e74c3c"
WARNING_COLOR = "#f39c12"
INFO_COLOR = "#3498db"

# Prediction thresholds
PLACEMENT_THRESHOLD = 0.5
HIGH_CONFIDENCE_THRESHOLD = 0.75
LOW_CONFIDENCE_THRESHOLD = 0.25
