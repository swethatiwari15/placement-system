"""
Student Input Form Component
Modern, organized form for student data collection
"""

import streamlit as st
from typing import Dict, Tuple
from utils.config import FEATURE_RANGES, FEATURE_LABELS, FEATURE_GROUPS


def render_student_form() -> Tuple[bool, Dict[str, float]]:
    """
    Render the student input form with organized sections.
    
    Returns:
        Tuple of (form_submitted, student_features_dict)
    """
    
    student_data = {}
    
    # Academic Profile Section
    st.markdown("### 📚 Academic Profile")
    st.markdown("*Your academic performance and foundation*")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        cgpa = st.slider(
            "CGPA",
            min_value=5.0,
            max_value=10.0,
            value=7.5,
            step=0.1,
            help="Cumulative Grade Point Average (out of 10)"
        )
        student_data['cgpa'] = cgpa
    
    with col2:
        st.empty()  # Placeholder for alignment
    
    st.markdown("---")
    
    # Experience & Projects Section
    st.markdown("### 💼 Experience & Projects")
    st.markdown("*Practical experience and hands-on work*")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        internships = st.number_input(
            "Number of Internships",
            min_value=0,
            max_value=5,
            value=1,
            step=1,
            help="Total number of internships completed"
        )
        student_data['internships'] = int(internships)
    
    with col2:
        projects = st.number_input(
            "Number of Projects",
            min_value=0,
            max_value=10,
            value=2,
            step=1,
            help="Total number of projects completed"
        )
        student_data['projects'] = int(projects)
    
    st.markdown("---")
    
    # Skills & Competencies Section
    st.markdown("### 🎯 Skills & Competencies")
    st.markdown("*Rate your skills on a scale of 1-10*")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        comm_skills = st.slider(
            "Communication Skills",
            min_value=1,
            max_value=10,
            value=6,
            help="Ability to communicate effectively"
        )
        student_data['communication_skills'] = int(comm_skills)
    
    with col2:
        problem_solving = st.slider(
            "Problem Solving",
            min_value=1,
            max_value=10,
            value=6,
            help="Ability to analyze and solve problems"
        )
        student_data['problem_solving'] = int(problem_solving)
    
    with col3:
        technical_skills = st.slider(
            "Technical Skills",
            min_value=1,
            max_value=10,
            value=6,
            help="Programming and technical proficiency"
        )
        student_data['technical_skills'] = int(technical_skills)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        leadership = st.slider(
            "Leadership",
            min_value=1,
            max_value=10,
            value=5,
            help="Ability to lead and take initiatives"
        )
        student_data['leadership'] = int(leadership)
    
    with col2:
        teamwork = st.slider(
            "Teamwork",
            min_value=1,
            max_value=10,
            value=6,
            help="Ability to work effectively in teams"
        )
        student_data['teamwork'] = int(teamwork)
    
    with col3:
        adaptability = st.slider(
            "Adaptability",
            min_value=1,
            max_value=10,
            value=6,
            help="Ability to adapt to new situations"
        )
        student_data['adaptability'] = int(adaptability)
    
    st.markdown("---")
    
    # Submit button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        submitted = st.button(
            "🚀 Predict Placement",
            use_container_width=True,
            type="primary"
        )
    
    return submitted, student_data


def render_form_summary(student_data: Dict[str, float]):
    """Render a summary of the submitted form."""
    
    st.markdown("### 📋 Student Profile Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="
            background: #f0f3f4;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        ">
            <div style="font-size: 12px; color: #666;">CGPA</div>
            <div style="font-size: 24px; font-weight: bold; color: #1f77b4;">
                {student_data.get('cgpa', 0):.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="
            background: #f0f3f4;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        ">
            <div style="font-size: 12px; color: #666;">Experience</div>
            <div style="font-size: 24px; font-weight: bold; color: #f39c12;">
                {student_data.get('internships', 0) + student_data.get('projects', 0)}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_skills = sum([
            student_data.get('communication_skills', 0),
            student_data.get('problem_solving', 0),
            student_data.get('technical_skills', 0),
            student_data.get('leadership', 0),
            student_data.get('teamwork', 0),
            student_data.get('adaptability', 0)
        ]) / 6
        
        st.markdown(f"""
        <div style="
            background: #f0f3f4;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        ">
            <div style="font-size: 12px; color: #666;">Avg Skills</div>
            <div style="font-size: 24px; font-weight: bold; color: #2ecc71;">
                {avg_skills:.1f}/10
            </div>
        </div>
        """, unsafe_allow_html=True)
