"""
Analytics Page
Model performance metrics and analysis
"""

import streamlit as st
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.components.cards import (
    render_metric_card, render_confusion_matrix, 
    render_metrics_table, render_feature_importance_chart
)
from utils.helpers import ModelManager, get_model_metrics, get_feature_importance


def render():
    """Render the analytics page."""
    
    st.markdown("# 📊 Analytics & Insights")
    
    st.markdown("""
    Explore model performance metrics, feature importance, and detailed 
    analysis of the placement prediction system.
    """)
    
    st.markdown("---")
    
    # Check if model is available
    manager = ModelManager()
    
    if not manager.is_available():
        st.warning("⚠️ Model is not trained yet. Please run the training script first.")
        return
    
    metrics = get_model_metrics()
    importance = get_feature_importance()
    
    if not metrics:
        st.info("ℹ️ No metrics available. Train the model first.")
        return
    
    # Model Performance Summary
    st.markdown("## 🎯 Model Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_metric_card(
            "Accuracy",
            f"{metrics.get('accuracy', 0)*100:.1f}%",
            "Overall correctness",
            "#1f77b4",
            "✓"
        )
    
    with col2:
        render_metric_card(
            "Precision",
            f"{metrics.get('precision', 0)*100:.1f}%",
            "True positive rate",
            "#f39c12",
            "🎯"
        )
    
    with col3:
        render_metric_card(
            "Recall",
            f"{metrics.get('recall', 0)*100:.1f}%",
            "Sensitivity",
            "#2ecc71",
            "📍"
        )
    
    with col4:
        render_metric_card(
            "F1-Score",
            f"{metrics.get('f1', 0):.4f}",
            "Harmonic mean",
            "#e74c3c",
            "⚡"
        )
    
    st.markdown("---")
    
    # Metrics Table
    st.markdown("## 📋 Detailed Metrics")
    render_metrics_table(metrics)
    
    st.markdown("---")
    
    # Confusion Matrix
    st.markdown("## 🔍 Confusion Matrix")
    
    if 'confusion_matrix' in metrics:
        render_confusion_matrix(metrics['confusion_matrix'])
        
        st.markdown("""
        **Interpretation:**
        - **True Negatives (TN):** Correctly predicted not placed
        - **False Positives (FP):** Incorrectly predicted placed
        - **False Negatives (FN):** Incorrectly predicted not placed  
        - **True Positives (TP):** Correctly predicted placed
        """)
    
    st.markdown("---")
    
    # Feature Importance
    st.markdown("## ⭐ Feature Importance Ranking")
    
    if importance:
        render_feature_importance_chart(importance)
        
        # Detailed importance table
        st.markdown("### Detailed Importance Scores")
        
        import pandas as pd
        importance_df = pd.DataFrame(
            list(importance.items()),
            columns=['Feature', 'Importance Score']
        ).sort_values('Importance Score', ascending=False)
        
        importance_df['Importance Score'] = importance_df['Importance Score'].apply(
            lambda x: f"{x:.4f}"
        )
        
        st.dataframe(importance_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        **What this means:**
        - Higher scores indicate greater influence on placement prediction
        - Features with high importance are strong indicators of placement potential
        - Focus on improving low-importance features for maximum impact on placement chances
        """)
    
    st.markdown("---")
    
    # Model Information
    st.markdown("## ℹ️ Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Algorithm**
        - Logistic Regression
        
        **Features Used**
        - 9 academic and skill-based features
        
        **Data Scaling**
        - StandardScaler normalization
        """)
    
    with col2:
        st.markdown("""
        **Model Type**
        - Binary Classification
        
        **Training Approach**
        - Train-test split: 80-20
        - Class balancing: Yes
        
        **Output**
        - Placement prediction (0: Not Placed, 1: Placed)
        - Probability score (0-1)
        """)
    
    st.markdown("---")
    
    # Recommendations
    st.markdown("## 💡 Using These Insights")
    
    st.markdown("""
    1. **For Students:**
       - Focus on improving high-importance features
       - Work on weak areas identified in your prediction
       - Use recommendations to increase placement chances
    
    2. **For Institutions:**
       - Understand which factors most influence placements
       - Design curriculum focusing on important skills
       - Monitor student development in key areas
    
    3. **For Placement Officers:**
       - Use model insights to identify at-risk students
       - Provide targeted support for weak areas
       - Track improvement trends over time
    """)


if __name__ == "__main__":
    render()
