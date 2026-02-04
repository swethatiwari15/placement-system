"""
AI Interview Page
Conducts interview with 5 HR questions and evaluates responses
"""

import streamlit as st
import json
from datetime import datetime
from pathlib import Path
import re

# Interview questions
INTERVIEW_QUESTIONS = [
    {
        "id": 1,
        "question": "Tell us about yourself and your background. What are your key strengths and what makes you a good fit for a placement?",
        "category": "Communication & Self-Awareness",
        "focus": "communication, confidence"
    },
    {
        "id": 2,
        "question": "Describe a challenging problem you solved recently, either in your projects or internships. What was your approach?",
        "category": "Problem-Solving",
        "focus": "problem-solving, technical thinking"
    },
    {
        "id": 3,
        "question": "Tell us about a time you worked in a team. How did you contribute, and how did you handle disagreements?",
        "category": "Teamwork & Collaboration",
        "focus": "teamwork, collaboration, soft skills"
    },
    {
        "id": 4,
        "question": "Where do you see yourself in 5 years? What are your career goals and how will this role help you achieve them?",
        "category": "Career Goals & Ambition",
        "focus": "career goals, ambition, clarity"
    },
    {
        "id": 5,
        "question": "Why should we hire you? What unique value do you bring to our organization?",
        "category": "Confidence & Value Proposition",
        "focus": "confidence, self-awareness, value"
    }
]

INTERVIEW_FILE = Path(__file__).parent.parent.parent / "data" / "interview_responses.json"
INTERVIEW_FILE.parent.mkdir(parents=True, exist_ok=True)


def initialize_interview_session():
    """Initialize interview session state."""
    if 'interview_started' not in st.session_state:
        st.session_state.interview_started = False
        st.session_state.current_question = 0
        st.session_state.responses = {}
        st.session_state.interview_complete = False


def save_interview_response(email: str, responses: dict, evaluation: dict):
    """Save interview responses and evaluation."""
    try:
        interview_data = {
            'email': email,
            'timestamp': datetime.now().isoformat(),
            'responses': responses,
            'evaluation': evaluation
        }
        
        data = []
        if INTERVIEW_FILE.exists():
            with open(INTERVIEW_FILE, 'r') as f:
                data = json.load(f)
        
        data.append(interview_data)
        
        with open(INTERVIEW_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        
        return True
    except Exception as e:
        st.error(f"Error saving interview: {str(e)}")
        return False


def evaluate_response(question_id: int, response_text: str) -> dict:
    """
    Evaluate interview response using simple NLP-based scoring.
    """
    question = next((q for q in INTERVIEW_QUESTIONS if q['id'] == question_id), None)
    if not question:
        return {}
    
    # Word count indicates effort
    word_count = len(response_text.split())
    
    # Check for positive indicators
    indicators = {
        'communication': ['clear', 'explain', 'understand', 'articulate', 'communicate', 'express', 'detail'],
        'confidence': ['confident', 'able', 'can', 'managed', 'led', 'achieved', 'successfully', 'strong'],
        'problem_solving': ['problem', 'solution', 'approach', 'logic', 'analysis', 'method', 'resolved', 'fixed'],
        'teamwork': ['team', 'collaborate', 'together', 'cooperate', 'support', 'help', 'worked', 'contributed'],
        'career_goals': ['career', 'goal', 'future', 'growth', 'learn', 'develop', 'opportunity', 'vision']
    }
    
    response_lower = response_text.lower()
    scores = {}
    
    # Score communication and clarity
    clarity_score = min(100, (word_count / 100) * 50)  # 50 points for detailed answer
    for keyword in indicators.get('communication', []):
        if keyword in response_lower:
            clarity_score += 10
    scores['communication_clarity'] = min(100, clarity_score)
    
    # Score confidence
    confidence_score = 40
    for keyword in indicators.get('confidence', []):
        if keyword in response_lower:
            confidence_score += 15
    scores['confidence_level'] = min(100, confidence_score)
    
    # Score problem-solving (for q2)
    if question_id == 2:
        ps_score = 40
        for keyword in indicators.get('problem_solving', []):
            if keyword in response_lower:
                ps_score += 15
        scores['problem_solving'] = min(100, ps_score)
    
    # Score teamwork (for q3)
    if question_id == 3:
        team_score = 40
        for keyword in indicators.get('teamwork', []):
            if keyword in response_lower:
                team_score += 15
        scores['teamwork'] = min(100, team_score)
    
    # Score career vision (for q4)
    if question_id == 4:
        career_score = 40
        for keyword in indicators.get('career_goals', []):
            if keyword in response_lower:
                career_score += 15
        scores['career_clarity'] = min(100, career_score)
    
    return scores


def render():
    """Render the interview page."""
    
    initialize_interview_session()
    
    st.markdown("# 🎤 AI Interview")
    
    # Check if student is registered
    if 'student_email' not in st.session_state:
        st.warning("⚠️ Please register first before starting the interview.")
        st.info("Go to 'Student Registration' page to register.")
        return
    
    student_name = st.session_state.get('student_name', 'Student')
    student_email = st.session_state.get('student_email', '')
    
    st.markdown(f"Welcome, **{student_name}**! 👋")
    
    st.markdown("""
    You'll answer 5 HR interview questions. This interview will evaluate your:
    - Communication and clarity
    - Confidence level
    - Problem-solving approach
    - Teamwork abilities
    - Career goals
    
    **Take your time and answer thoughtfully. No perfect answers—just be yourself!**
    """)
    
    st.markdown("---")
    
    if st.session_state.interview_complete:
        st.success("✓ Interview Complete!")
        st.info("Click on 'Interview Feedback' from the sidebar to see your results.")
        return
    
    # Interview progress
    progress = len(st.session_state.responses) / len(INTERVIEW_QUESTIONS)
    st.progress(progress)
    st.markdown(f"**Progress: {len(st.session_state.responses)}/{len(INTERVIEW_QUESTIONS)} questions answered**")
    
    st.markdown("---")
    
    if not st.session_state.interview_started:
        if st.button("🎯 Start Interview", use_container_width=True, type="primary"):
            st.session_state.interview_started = True
            st.rerun()
    else:
        # Get current question
        current_q_index = len(st.session_state.responses)
        
        if current_q_index < len(INTERVIEW_QUESTIONS):
            question_data = INTERVIEW_QUESTIONS[current_q_index]
            
            # Display question
            st.markdown(f"### Question {question_data['id']}/{len(INTERVIEW_QUESTIONS)}")
            st.markdown(f"**{question_data['question']}**")
            st.markdown(f"*{question_data['category']}*")
            st.markdown("---")
            
            # Response input
            response = st.text_area(
                "Your Answer",
                placeholder="Type your response here... (minimum 50 characters)",
                height=150,
                label_visibility="collapsed"
            )
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("📝 Submit Answer", use_container_width=True, type="primary"):
                    if len(response.strip()) < 50:
                        st.error("❌ Please provide a more detailed answer (at least 50 characters)")
                    else:
                        # Evaluate response
                        evaluation = evaluate_response(question_data['id'], response)
                        
                        # Store response
                        st.session_state.responses[question_data['id']] = {
                            'question': question_data['question'],
                            'answer': response,
                            'evaluation': evaluation,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        st.success("✓ Answer submitted!")
                        
                        # Check if all questions answered
                        if len(st.session_state.responses) == len(INTERVIEW_QUESTIONS):
                            # Save interview and move to feedback
                            overall_evaluation = calculate_overall_evaluation(st.session_state.responses)
                            save_interview_response(student_email, st.session_state.responses, overall_evaluation)
                            st.session_state.interview_complete = True
                            st.info("All questions answered! Moving to feedback...")
                            st.rerun()
                        else:
                            st.rerun()
            
            with col2:
                if st.button("⏭️ Next Question", use_container_width=True):
                    if not response or len(response.strip()) < 50:
                        st.error("❌ Please provide a detailed answer before proceeding")
                    else:
                        # Evaluate response
                        evaluation = evaluate_response(question_data['id'], response)
                        
                        # Store response
                        st.session_state.responses[question_data['id']] = {
                            'question': question_data['question'],
                            'answer': response,
                            'evaluation': evaluation,
                            'timestamp': datetime.now().isoformat()
                        }
                        st.rerun()


def calculate_overall_evaluation(responses: dict) -> dict:
    """Calculate overall evaluation from all responses."""
    all_scores = {}
    
    for q_id, response_data in responses.items():
        evaluation = response_data.get('evaluation', {})
        for metric, score in evaluation.items():
            if metric not in all_scores:
                all_scores[metric] = []
            all_scores[metric].append(score)
    
    # Average scores
    overall = {}
    for metric, scores in all_scores.items():
        overall[metric] = sum(scores) / len(scores) if scores else 0
    
    return overall


if __name__ == "__main__":
    render()
