"""
Prediction Page
Student input form and prediction display
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.components.form import render_student_form, render_form_summary
from app.components.cards import render_prediction_result, render_feature_comparison, render_insights
from utils.helpers import ModelManager, validate_student_input, get_placement_insights
from utils.config import FEATURE_RANGES, FEATURE_LABELS


def render():
    """Render the prediction page."""
    
    st.markdown("# 🔮 Student Placement Prediction")
    
    st.markdown("""
    Enter your academic profile, experience, and skills to get an instant 
    placement prediction with detailed insights.
    """)
    
    st.markdown("---")
    
    # Render form
    submitted, student_data = render_student_form()
    
    if submitted:
        # Validate input
        is_valid, error_msg = validate_student_input(student_data)
        
        if not is_valid:
            st.error(f"❌ Validation Error: {error_msg}")
            return
        
        # Get prediction
        st.markdown("---")
        st.markdown("## 📊 Prediction Results")
        
        try:
            manager = ModelManager()
            prediction_result = manager.predict(student_data)
            
            # Display form summary
            render_form_summary(student_data)
            
            st.markdown("---")
            
            # Display prediction
            render_prediction_result(
                placed=prediction_result['placed'],
                probability=prediction_result['probability'],
                confidence_level=prediction_result['confidence_level'],
                features=student_data
            )
            
            st.markdown("---")
            
            # Display feature comparison
            st.markdown("## 📈 Performance Analysis")
            render_feature_comparison(student_data, FEATURE_RANGES, FEATURE_LABELS)
            
            st.markdown("---")
            
            # Display insights
            st.markdown("## 💡 Key Insights")
            insights = get_placement_insights(student_data, prediction_result)
            render_insights(insights)
            
            # Success message
            st.success("✓ Prediction completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.info("Please check the Analytics page to ensure the model is properly trained.")


if __name__ == "__main__":
    render()
