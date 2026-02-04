"""
Home Page
Overview and introduction to the system
"""

import streamlit as st


def render():
    """Render the home page."""
    
    st.markdown("""
    # 🎓 Student Placement Prediction System
    
    Welcome to the **Student Placement Prediction System** — a modern machine learning application 
    designed to help students understand their placement potential.
    """)
    
    # Overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        ">
            <div style="font-size: 48px; margin-bottom: 10px;">🤖</div>
            <h3 style="color: white; border: none;">AI Powered</h3>
            <p>Logistic Regression model trained on real placement data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        ">
            <div style="font-size: 48px; margin-bottom: 10px;">📊</div>
            <h3 style="color: white; border: none;">Data Driven</h3>
            <p>Detailed analytics and insights into your profile</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        ">
            <div style="font-size: 48px; margin-bottom: 10px;">✨</div>
            <h3 style="color: white; border: none;">Modern Design</h3>
            <p>Clean, intuitive interface for easy navigation</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features
    st.markdown("## 🌟 Key Features")
    
    st.markdown("""
    **📝 Student Registration & Interview**
    - Register with your profile information
    - Answer 5 HR interview questions
    - Get AI-powered evaluation of your responses
    - Receive detailed feedback and improvement suggestions
    
    **🔮 Student Prediction**
    - Input your academic profile, experience, and skills
    - Get instant placement prediction with probability scores
    - Receive personalized recommendations
    
    **📈 Analytics & Insights**
    - View model performance metrics
    - Understand feature importance
    - Analyze prediction patterns
    
    **💡 Explainability**
    - Understand why you're placed or not
    - Identify your strong factors
    - Get actionable recommendations
    
    **🔐 Production Ready**
    - Professional-grade ML pipeline
    - Data validation and error handling
    - Scalable architecture
    """)
    
    st.markdown("---")
    
    # How it works
    st.markdown("## 🚀 How It Works")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Step 1: Input Your Profile
        Fill in your academic details, experience, and skills through our 
        organized form with:
        - Academic Performance (CGPA)
        - Experience (Internships, Projects)
        - Skills (Communication, Technical, Leadership, etc.)
        """)
    
    with col2:
        st.markdown("""
        ### Step 2: Get Prediction
        Our ML model analyzes your profile and provides:
        - Placement prediction (Placed / Not Placed)
        - Confidence score
        - Probability percentage
        """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Step 3: Receive Insights
        Get detailed analysis including:
        - Strong factors in your profile
        - Areas needing improvement
        - Personalized recommendations
        """)
    
    with col2:
        st.markdown("""
        ### Step 4: Take Action
        Use insights to:
        - Improve weak areas
        - Leverage your strengths
        - Increase placement chances
        """)
    
    st.markdown("---")
    
    # Call to action
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 40px;
        border-radius: 10px;
        text-align: center;
        margin: 30px 0;
    ">
        <h2 style="color: white; border: none;">Ready to Check Your Placement Chances?</h2>
        <p style="font-size: 16px; color: #ddd;">
            Use our AI-powered predictor to get instant insights into your placement potential.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # FAQ
    st.markdown("## ❓ Frequently Asked Questions")
    
    with st.expander("How accurate is the prediction?"):
        st.write("""
        Our model achieves high accuracy through proper data preprocessing, 
        feature scaling, and cross-validation. Check the Analytics page for 
        detailed performance metrics.
        """)
    
    with st.expander("What factors influence placement the most?"):
        st.write("""
        The model considers all factors: academic performance (CGPA), practical 
        experience (internships and projects), and soft skills. Check the 
        Analytics page to see feature importance rankings.
        """)
    
    with st.expander("How can I improve my placement chances?"):
        st.write("""
        Use our personalized recommendations after getting your prediction. 
        The system identifies your weak areas and suggests specific improvements.
        """)
    
    with st.expander("Is my data secure?"):
        st.write("""
        All predictions are performed locally without storing personal data. 
        Your information is not saved or shared.
        """)


if __name__ == "__main__":
    render()
