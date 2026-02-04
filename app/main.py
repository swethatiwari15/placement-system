"""
Main Streamlit Application
Multi-page Student Placement Prediction System
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pages import home, prediction, analytics, about, registration, interview, feedback
from utils.helpers import ModelManager

# Page configuration
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        color: white;
        font-weight: 600;
    }
    
    [data-testid="stSidebar"] [role="radiogroup"] > label {
        color: white;
    }
    
    .main {
        background: #f8f9fa;
    }
    
    h1 {
        color: #1f77b4;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 10px;
    }
    
    h2 {
        color: #333;
        margin-top: 20px;
    }
    
    .metric-container {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 10px 30px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.markdown("""
<div style="text-align: center; color: white; padding: 20px 0;">
    <h1 style="color: white; border: none; font-size: 28px;">🎓 SPP</h1>
    <p style="color: #ddd; font-size: 12px;">Student Placement Predictor</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Check model availability
model_manager = ModelManager()
model_available = model_manager.is_available()

if not model_available:
    st.sidebar.warning("⚠️ Model not trained. Please run the training script first.")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    options=[
        "Home",
        "Student Registration",
        "Student Interview",
        "Interview Feedback",
        "Student Prediction",
        "Analytics & Insights",
        "About & Methodology"
    ],
    label_visibility="collapsed"
)

# Route to pages
if page == "Home":
    st.sidebar.markdown("# Home")
    home.render()
elif page == "Student Registration":
    st.sidebar.markdown("# Student Registration")
    registration.render()
elif page == "Student Interview":
    st.sidebar.markdown("# Student Interview")
    interview.render()
elif page == "Interview Feedback":
    st.sidebar.markdown("# Interview Feedback")
    feedback.render()
elif page == "Student Prediction":
    st.sidebar.markdown("# Student Prediction")
    if model_available:
        prediction.render()
    else:
        st.error("Model is not available. Please train the model first.")
elif page == "Analytics & Insights":
    st.sidebar.markdown("# Analytics & Insights")
    analytics.render()
elif page == "About & Methodology":
    st.sidebar.markdown("# About & Methodology")
    about.render()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; font-size: 12px; padding: 20px 0;">
    <p>Student Placement Prediction System v1.0</p>
    <p>Built with Streamlit & Machine Learning</p>
    <p>© 2026 All rights reserved</p>
</div>
""", unsafe_allow_html=True)
