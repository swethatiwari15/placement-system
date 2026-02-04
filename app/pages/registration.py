"""
Student Registration Page
Handles student registration before interview
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
import re
from datetime import datetime

# Create data directory if not exists
REGISTRATION_FILE = Path(__file__).parent.parent.parent / "data" / "students_registered.csv"
REGISTRATION_FILE.parent.mkdir(parents=True, exist_ok=True)


def email_exists(email: str) -> bool:
    """Check if email already exists in registration."""
    if not REGISTRATION_FILE.exists():
        return False
    df = pd.read_csv(REGISTRATION_FILE)
    return email.lower() in df['email'].str.lower().values


def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_phone(phone: str) -> bool:
    """Validate phone number format."""
    phone_clean = ''.join(filter(str.isdigit, phone))
    return len(phone_clean) >= 10


def save_registration(student_data: dict) -> bool:
    """Save student registration to CSV."""
    try:
        student_data['registration_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        df_new = pd.DataFrame([student_data])
        
        if REGISTRATION_FILE.exists():
            df_existing = pd.read_csv(REGISTRATION_FILE)
            df = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df = df_new
        
        df.to_csv(REGISTRATION_FILE, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving registration: {str(e)}")
        return False


def render():
    """Render the registration page."""
    
    st.markdown("# 📝 Student Registration")
    
    st.markdown("""
    Register to participate in our AI-powered interview process.
    After registration, you'll get instant access to the interview.
    """)
    
    st.markdown("---")
    
    # Check if already registered
    if 'registration_complete' not in st.session_state:
        st.session_state.registration_complete = False
    
    if st.session_state.registration_complete:
        st.success("✓ Registration successful! Redirecting to interview...")
        st.info("Click 'Student Interview' from the sidebar to start your interview.")
        return
    
    # Registration form
    with st.form("registration_form"):
        
        # Personal Information
        st.markdown("### 👤 Personal Information")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            name = st.text_input(
                "Full Name *",
                placeholder="John Doe",
                help="Your full name"
            )
        with col2:
            email = st.text_input(
                "Email Address *",
                placeholder="john@example.com",
                help="Your email address"
            )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            phone = st.text_input(
                "Phone Number *",
                placeholder="+1 (555) 123-4567",
                help="Your contact number"
            )
        with col2:
            department = st.selectbox(
                "Department *",
                options=[
                    "Computer Science",
                    "Electronics",
                    "Electrical",
                    "Mechanical",
                    "Civil",
                    "Chemical",
                    "Other"
                ],
                help="Your department"
            )
        
        st.markdown("---")
        
        # Academic Information
        st.markdown("### 📚 Academic Information")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            cgpa = st.slider(
                "CGPA *",
                min_value=5.0,
                max_value=10.0,
                value=7.5,
                step=0.1
            )
        with col2:
            semester = st.number_input(
                "Current Semester *",
                min_value=1,
                max_value=8,
                value=6,
                step=1
            )
        
        st.markdown("---")
        
        # Skills
        st.markdown("### 🎯 Skills")
        st.markdown("*Select your technical skills*")
        
        tech_skills = st.multiselect(
            "Technical Skills *",
            options=[
                "Python",
                "Java",
                "C++",
                "JavaScript",
                "SQL",
                "Web Development",
                "Data Science",
                "Machine Learning",
                "Cloud Computing",
                "DevOps",
                "Mobile Development",
                "UI/UX Design"
            ],
            help="Select all applicable technical skills"
        )
        
        st.markdown("*Rate your soft skills (1-10)*")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            communication = st.slider(
                "Communication *",
                min_value=1,
                max_value=10,
                value=6
            )
        with col2:
            teamwork = st.slider(
                "Teamwork *",
                min_value=1,
                max_value=10,
                value=6
            )
        with col3:
            leadership = st.slider(
                "Leadership *",
                min_value=1,
                max_value=10,
                value=5
            )
        
        st.markdown("---")
        
        # Submit button
        submitted = st.form_submit_button(
            "🚀 Register & Start Interview",
            use_container_width=True,
            type="primary"
        )
    
    if submitted:
        # Validation
        errors = []
        
        if not name or len(name.strip()) < 2:
            errors.append("Name must be at least 2 characters")
        
        if not email or not validate_email(email):
            errors.append("Invalid email format")
        
        if email_exists(email):
            errors.append("This email is already registered")
        
        if not phone or not validate_phone(phone):
            errors.append("Invalid phone number (need at least 10 digits)")
        
        if not tech_skills:
            errors.append("Please select at least one technical skill")
        
        if errors:
            st.error("❌ Registration Failed:")
            for error in errors:
                st.write(f"  • {error}")
        else:
            # Save registration
            student_data = {
                'name': name.strip(),
                'email': email.strip().lower(),
                'phone': phone.strip(),
                'department': department,
                'cgpa': cgpa,
                'semester': int(semester),
                'communication_skills': int(communication),
                'teamwork': int(teamwork),
                'leadership': int(leadership),
                'technical_skills': '|'.join(tech_skills)
            }
            
            if save_registration(student_data):
                st.session_state.registration_complete = True
                st.session_state.student_email = email.strip().lower()
                st.session_state.student_name = name.strip()
                st.success("✓ Registration successful!")
                st.rerun()
            else:
                st.error("Failed to save registration. Please try again.")


if __name__ == "__main__":
    render()
