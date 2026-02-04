"""
About & Methodology Page
Technical documentation and system information
"""

import streamlit as st


def render():
    """Render the about page."""
    
    st.markdown("# 📖 About & Methodology")
    
    st.markdown("""
    Learn about the technical implementation, ML methodology, and system architecture 
    of the Student Placement Prediction System.
    """)
    
    st.markdown("---")
    
    # Overview
    st.markdown("## 🎯 System Overview")
    
    st.markdown("""
    The Student Placement Prediction System is a production-ready machine learning 
    application that predicts whether a student will be placed in their placement drive 
    based on their academic profile, experience, and skills.
    
    **Key Characteristics:**
    - **Technology Stack:** Python, Scikit-Learn, Streamlit
    - **Model:** Logistic Regression with StandardScaler
    - **Features:** 9 quantitative features
    - **Output:** Binary classification with probability scores
    """)
    
    st.markdown("---")
    
    # Architecture
    st.markdown("## 🏗️ System Architecture")
    
    st.markdown("""
    ### Project Structure
    
    ```
    spp-project/
    ├── data/
    │   ├── placement_data.csv          # Dataset
    │   └── preprocess.py               # Data preprocessing
    ├── models/
    │   ├── train.py                    # Model training pipeline
    │   ├── placement_model.pkl         # Trained model
    │   ├── placement_model_scaler.pkl  # Feature scaler
    │   ├── placement_model_metrics.json # Performance metrics
    │   └── placement_model_importance.json # Feature importance
    ├── app/
    │   ├── main.py                     # Main Streamlit app
    │   ├── pages/
    │   │   ├── home.py                 # Home page
    │   │   ├── prediction.py           # Prediction page
    │   │   ├── analytics.py            # Analytics page
    │   │   └── about.py                # About page
    │   ├── components/
    │   │   ├── cards.py                # UI card components
    │   │   └── form.py                 # Student input form
    │   └── styles/
    │       └── custom.css              # Custom styling
    ├── utils/
    │   ├── config.py                   # Configuration & constants
    │   ├── helpers.py                  # Helper functions
    │   └── __init__.py
    ├── scripts/
    │   └── train_model.py              # Training script
    ├── docs/
    │   └── METHODOLOGY.md              # Technical documentation
    └── README.md                        # Project documentation
    ```
    """)
    
    st.markdown("---")
    
    # ML Pipeline
    st.markdown("## 🔄 Machine Learning Pipeline")
    
    st.markdown("""
    ### 1. Data Preparation
    - **Source:** Student placement dataset with 60 records
    - **Features:** 9 features (CGPA, internships, projects, communication skills, etc.)
    - **Target:** Binary (Placed: 1, Not Placed: 0)
    - **Validation:** Check for missing values, duplicates, and data types
    
    ### 2. Feature Engineering
    - **Feature Scaling:** StandardScaler normalization
      - Transforms features to have mean=0 and std=1
      - Important for Logistic Regression convergence
    - **No feature selection:** All 9 features are used
    
    ### 3. Model Training
    - **Algorithm:** Logistic Regression
      - Binary classifier suitable for placement prediction
      - Provides probability scores (0-1)
      - Interpretable coefficients for feature importance
    
    - **Hyperparameters:**
      - Max iterations: 1000
      - Solver: LBFGS
      - Class weight: Balanced (handles imbalanced classes)
    
    ### 4. Validation Strategy
    - **Train-Test Split:** 80% train, 20% test
    - **Stratification:** Maintains class distribution
    - **Cross-Validation:** Ensures model generalization
    
    ### 5. Model Evaluation
    - **Metrics Computed:**
      - Accuracy: Overall correctness
      - Precision: True positive rate among predicted positives
      - Recall: True positive rate among actual positives
      - F1-Score: Harmonic mean of precision and recall
      - AUC-ROC: Area under the ROC curve
      - Confusion Matrix: Detailed classification breakdown
    
    ### 6. Model Deployment
    - **Serialization:** Model and scaler saved as pickle files
    - **Metadata:** Metrics and feature importance stored as JSON
    - **Accessibility:** Easy loading for inference on new data
    """)
    
    st.markdown("---")
    
    # Features
    st.markdown("## 📊 Feature Description")
    
    features_info = {
        "CGPA": {
            "range": "5.0 - 10.0",
            "description": "Cumulative Grade Point Average, primary academic indicator",
            "importance": "Very High"
        },
        "Internships": {
            "range": "0 - 5",
            "description": "Number of internships completed, shows practical experience",
            "importance": "High"
        },
        "Projects": {
            "range": "0 - 10",
            "description": "Number of projects completed, demonstrates hands-on skills",
            "importance": "High"
        },
        "Communication Skills": {
            "range": "1 - 10",
            "description": "Ability to communicate effectively (self-rated)",
            "importance": "Medium"
        },
        "Problem Solving": {
            "range": "1 - 10",
            "description": "Ability to analyze and solve problems (self-rated)",
            "importance": "High"
        },
        "Technical Skills": {
            "range": "1 - 10",
            "description": "Programming and technical proficiency (self-rated)",
            "importance": "Very High"
        },
        "Leadership": {
            "range": "1 - 10",
            "description": "Leadership and initiative-taking abilities (self-rated)",
            "importance": "Medium"
        },
        "Teamwork": {
            "range": "1 - 10",
            "description": "Ability to work effectively in teams (self-rated)",
            "importance": "Medium"
        },
        "Adaptability": {
            "range": "1 - 10",
            "description": "Ability to adapt to new situations (self-rated)",
            "importance": "Medium"
        }
    }
    
    for feature, info in features_info.items():
        with st.expander(f"🔹 {feature}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Range:** {info['range']}")
            with col2:
                st.markdown(f"**Importance:** {info['importance']}")
            st.markdown(f"**Description:** {info['description']}")
    
    st.markdown("---")
    
    # Prediction Logic
    st.markdown("## 🧠 Prediction Logic")
    
    st.markdown("""
    ### How Predictions are Made
    
    1. **Input Normalization:**
       - Student features are scaled using the fitted StandardScaler
       - This ensures consistency with training data
    
    2. **Model Inference:**
       - Scaled features are passed to the trained Logistic Regression model
       - Model computes prediction probability (0-1)
    
    3. **Classification:**
       - If probability > 0.5: Predicted as "Placed"
       - If probability ≤ 0.5: Predicted as "Not Placed"
    
    4. **Confidence Assessment:**
       - Distance from threshold (0.5) determines confidence level
       - High confidence: > 0.75
       - Medium confidence: 0.25 - 0.75
       - Low confidence: < 0.25
    
    ### Feature Importance Extraction
    
    - **Method:** Coefficient-based importance from Logistic Regression
    - **Calculation:** 
      - Take absolute values of model coefficients
      - Normalize by total sum
      - Higher values = greater influence on placement prediction
    """)
    
    st.markdown("---")
    
    # Explainability
    st.markdown("## 💡 Model Explainability")
    
    st.markdown("""
    ### Why Features Matter
    
    The system explains predictions by:
    
    1. **Strong Factors:** Features where student performs above 70th percentile
       - Highlight competitive advantages
       - Indicate readiness for placement
    
    2. **Weak Areas:** Features where student performs below 40th percentile
       - Identify improvement opportunities
       - Suggest targeted development
    
    3. **Recommendations:** Personalized action items based on:
       - Prediction outcome
       - Feature performance
       - Industry standards
    
    ### Confidence Levels
    
    - **High Confidence:** Model is very sure about the prediction
      - Reflects clear performance patterns
      - Reliable for decision-making
    
    - **Medium Confidence:** Model has moderate certainty
      - Mixed signals in the profile
      - Student could go either way with effort
    
    - **Low Confidence:** Model has low certainty
      - Features are borderline
      - Small improvements could change outcome
    """)
    
    st.markdown("---")
    
    # Performance Metrics
    st.markdown("## 📈 Understanding Model Metrics")
    
    metric_explanations = {
        "Accuracy": "Percentage of correct predictions (both placed and not placed)",
        "Precision": "Of students predicted as placed, how many actually got placed",
        "Recall": "Of students who actually got placed, how many were correctly identified",
        "F1-Score": "Balanced measure combining precision and recall (0-1 scale)",
        "AUC-ROC": "Measures performance across all classification thresholds (0-1 scale)",
        "Confusion Matrix": "Breakdown of correct/incorrect predictions for each class"
    }
    
    for metric, explanation in metric_explanations.items():
        st.markdown(f"**{metric}:** {explanation}")
    
    st.markdown("---")
    
    # Limitations
    st.markdown("## ⚠️ Limitations & Considerations")
    
    st.markdown("""
    1. **Data Size:** Model trained on 60 records. Performance may vary with larger datasets.
    
    2. **Feature Self-Rating:** Soft skills (communication, problem-solving, etc.) are self-assessed,
       which may have inherent bias.
    
    3. **Context:** Model doesn't consider:
       - Job market conditions
       - Company-specific requirements
       - Interview performance
       - Personal circumstances
    
    4. **Generalization:** Predictions are based on historical patterns and may not be
       100% accurate for individual cases.
    
    5. **Fairness:** Model should be used as a guide, not the sole decision-making tool.
    """)
    
    st.markdown("---")
    
    # Contact & Support
    st.markdown("## 📞 Support & Contact")
    
    st.markdown("""
    For questions, suggestions, or feedback about the system, please reach out through:
    
    - **Email:** support@spp.edu
    - **Documentation:** Check the `/docs` folder
    - **GitHub:** [Project Repository]
    - **Issues:** Report bugs and request features
    
    **Last Updated:** January 2026  
    **Version:** 1.0.0  
    **Status:** Production Ready ✓
    """)


if __name__ == "__main__":
    render()
