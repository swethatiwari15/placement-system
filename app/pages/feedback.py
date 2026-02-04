"""
Interview Feedback Page
Shows evaluation results and improvement suggestions
"""

import streamlit as st
import json
from pathlib import Path
import pandas as pd

INTERVIEW_FILE = Path(__file__).parent.parent.parent / "data" / "interview_responses.json"


def get_student_interview_data(email: str):
    """Get the most recent interview data for a student."""
    if not INTERVIEW_FILE.exists():
        return None
    
    try:
        with open(INTERVIEW_FILE, 'r') as f:
            all_data = json.load(f)
        
        # Get most recent interview for this email
        student_interviews = [d for d in all_data if d['email'] == email]
        if student_interviews:
            return student_interviews[-1]
    except:
        pass
    
    return None


def get_strengths(evaluation: dict) -> list:
    """Generate strengths from evaluation scores."""
    strengths = []
    
    if evaluation.get('communication_clarity', 0) >= 70:
        strengths.append("Strong communication and ability to articulate ideas clearly")
    
    if evaluation.get('confidence_level', 0) >= 70:
        strengths.append("Good confidence level and positive attitude")
    
    if evaluation.get('problem_solving', 0) >= 70:
        strengths.append("Strong problem-solving and analytical thinking")
    
    if evaluation.get('teamwork', 0) >= 70:
        strengths.append("Good teamwork and collaboration skills")
    
    if evaluation.get('career_clarity', 0) >= 70:
        strengths.append("Clear career goals and vision for the future")
    
    # Fallback if no scores are high
    if not strengths:
        strengths.append("Good effort and willingness to learn")
    
    return strengths


def get_weak_areas(evaluation: dict) -> list:
    """Generate weak areas from evaluation scores."""
    weak_areas = []
    
    if evaluation.get('communication_clarity', 0) < 60:
        weak_areas.append("Communication clarity")
    
    if evaluation.get('confidence_level', 0) < 60:
        weak_areas.append("Confidence and assertiveness")
    
    if evaluation.get('problem_solving', 0) < 60:
        weak_areas.append("Problem-solving approach")
    
    if evaluation.get('teamwork', 0) < 60:
        weak_areas.append("Teamwork and collaboration")
    
    if evaluation.get('career_clarity', 0) < 60:
        weak_areas.append("Career vision and goals")
    
    return weak_areas


def get_suggestions(evaluation: dict, student_data: dict = None) -> dict:
    """Generate improvement suggestions."""
    suggestions = {
        'communication': [],
        'technical': [],
        'confidence': [],
        'interview_behavior': []
    }
    
    comm_score = evaluation.get('communication_clarity', 0)
    if comm_score < 70:
        suggestions['communication'].extend([
            "Practice explaining your ideas more clearly and concisely",
            "Use specific examples and avoid vague statements",
            "Work on storytelling - structure your answers with context, challenge, and resolution"
        ])
    
    confidence_score = evaluation.get('confidence_level', 0)
    if confidence_score < 70:
        suggestions['confidence'].extend([
            "Take deep breaths before answering - it helps with composure",
            "Speak with conviction - avoid phrases like 'I think' or 'maybe'",
            "Practice mock interviews to build confidence",
            "Highlight your achievements and accomplishments"
        ])
    
    if evaluation.get('problem_solving', 0) < 70:
        suggestions['technical'].extend([
            "Learn to break down complex problems into smaller parts",
            "Practice explaining your thought process while solving problems",
            "Study real-world case studies and solutions"
        ])
    
    suggestions['interview_behavior'].extend([
        "Always answer with specific examples, not just generic statements",
        "Research the company before interviews",
        "Prepare 2-3 questions to ask the interviewer",
        "Practice active listening and don't interrupt",
        "Follow up with a thank you email after interviews"
    ])
    
    return suggestions


def render():
    """Render the feedback page."""
    
    st.markdown("# 📊 Interview Feedback")
    
    # Check if student is registered
    if 'student_email' not in st.session_state:
        st.warning("⚠️ Please register and complete the interview first.")
        return
    
    student_email = st.session_state.get('student_email', '')
    student_name = st.session_state.get('student_name', 'Student')
    
    # Get interview data
    interview_data = get_student_interview_data(student_email)
    
    if not interview_data:
        st.info("📋 No completed interview found. Please complete the interview first.")
        return
    
    st.success(f"✓ Interview completed on {interview_data['timestamp'][:10]}")
    st.markdown(f"**Student:** {student_name}")
    st.markdown("---")
    
    evaluation = interview_data['evaluation']
    responses = interview_data['responses']
    
    # Overall scores
    st.markdown("## 📈 Your Evaluation Scores")
    
    score_cols = st.columns(5)
    
    scores_to_display = [
        ('Communication', 'communication_clarity'),
        ('Confidence', 'confidence_level'),
        ('Problem Solving', 'problem_solving'),
        ('Teamwork', 'teamwork'),
        ('Career Clarity', 'career_clarity')
    ]
    
    for i, (label, key) in enumerate(scores_to_display):
        with score_cols[i]:
            score = evaluation.get(key, 0)
            if score > 0:
                # Simple visual representation
                percentage = int(score)
                bar_fill = "█" * (percentage // 10) + "░" * (10 - percentage // 10)
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; background: #f0f2f6; border-radius: 10px;">
                    <div style="font-weight: bold; font-size: 14px;">{label}</div>
                    <div style="font-size: 20px; font-weight: bold; color: #667eea;">{percentage}</div>
                    <div style="font-size: 10px; color: #999;">{'█' * (percentage // 20)}{'░' * (5 - percentage // 20)}</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Strengths
    st.markdown("## 💪 Your Strengths")
    strengths = get_strengths(evaluation)
    for i, strength in enumerate(strengths, 1):
        st.markdown(f"✓ {strength}")
    
    st.markdown("---")
    
    # Areas to improve
    weak_areas = get_weak_areas(evaluation)
    if weak_areas:
        st.markdown("## 🎯 Areas to Improve")
        for i, area in enumerate(weak_areas, 1):
            st.markdown(f"• {area}")
        st.markdown("---")
    
    # Suggestions
    st.markdown("## 💡 Suggestions for Improvement")
    
    suggestions = get_suggestions(evaluation)
    
    if suggestions['communication']:
        st.markdown("**Communication Skills**")
        for suggestion in suggestions['communication']:
            st.markdown(f"  • {suggestion}")
        st.markdown("")
    
    if suggestions['confidence']:
        st.markdown("**Build Your Confidence**")
        for suggestion in suggestions['confidence']:
            st.markdown(f"  • {suggestion}")
        st.markdown("")
    
    if suggestions['technical']:
        st.markdown("**Technical & Problem-Solving**")
        for suggestion in suggestions['technical']:
            st.markdown(f"  • {suggestion}")
        st.markdown("")
    
    if suggestions['interview_behavior']:
        st.markdown("**Interview Best Practices**")
        for suggestion in suggestions['interview_behavior']:
            st.markdown(f"  • {suggestion}")
    
    st.markdown("---")
    
    # Detailed response review
    with st.expander("📝 View Your Detailed Responses"):
        for q_id in sorted(responses.keys()):
            response_data = responses[q_id]
            st.markdown(f"**Q{q_id}: {response_data['question'][:80]}...**")
            st.markdown(f"*Your Answer:*")
            st.markdown(f"> {response_data['answer']}")
            
            eval_data = response_data.get('evaluation', {})
            if eval_data:
                st.markdown(f"*Evaluation:*")
                for metric, score in eval_data.items():
                    metric_label = metric.replace('_', ' ').title()
                    st.markdown(f"  - {metric_label}: {int(score)}/100")
            st.markdown("---")
    
    # Final message
    st.markdown("---")
    st.markdown("""
    ### 🚀 Next Steps
    
    1. **Practice** - Work on the weak areas identified above
    2. **Prepare** - Research companies and practice more interviews
    3. **Apply** - Use these insights in your actual job interviews
    4. **Improve** - Take the interview again anytime to track your progress
    
    Good luck with your placement journey! 🎓
    """)


if __name__ == "__main__":
    render()
