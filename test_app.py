#!/usr/bin/env python
"""
Test script to verify the application is working correctly
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("=" * 70)
    print("STUDENT PLACEMENT PREDICTION - APPLICATION TEST")
    print("=" * 70)
    
    # Test 1: Module imports
    print("\n[TEST 1] Testing module imports...")
    try:
        from app.pages import home, prediction, analytics, about, registration, interview, feedback
        from app.components import form, cards
        from utils import config, helpers
        from data.preprocess import DataPreprocessor
        from models.train import ModelPipeline
        print("✓ All modules imported successfully")
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Test 2: Model loading
    print("\n[TEST 2] Loading trained model...")
    try:
        from utils.helpers import ModelManager
        manager = ModelManager()
        if not manager.is_available():
            print("✗ Model not available")
            return False
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False
    
    # Test 3: Data preprocessing
    print("\n[TEST 3] Testing data preprocessing...")
    try:
        preprocessor = DataPreprocessor("data/placement_data.csv")
        df = preprocessor.load_data()
        X, y = preprocessor.get_features_and_target()
        summary = preprocessor.get_data_summary()
        print(f"✓ Data loaded: {df.shape[0]} records")
        print(f"  - Features: {summary['features']}")
        print(f"  - Placement rate: {summary['placement_rate']:.1f}%")
    except Exception as e:
        print(f"✗ Error in data preprocessing: {e}")
        return False
    
    # Test 4: Prediction pipeline
    print("\n[TEST 4] Testing prediction pipeline...")
    try:
        from utils.helpers import validate_student_input, get_placement_insights
        
        test_features = {
            'cgpa': 8.5,
            'internships': 2,
            'projects': 4,
            'communication_skills': 8,
            'problem_solving': 9,
            'technical_skills': 8,
            'leadership': 7,
            'teamwork': 8,
            'adaptability': 8
        }
        
        # Validate input
        is_valid, msg = validate_student_input(test_features)
        if not is_valid:
            print(f"✗ Validation failed: {msg}")
            return False
        
        # Make prediction
        result = manager.predict(test_features)
        insights = get_placement_insights(test_features, result)
        
        print("✓ Prediction successful")
        print(f"  - Status: {'Placed' if result['placed'] else 'Not Placed'}")
        print(f"  - Probability: {result['probability']*100:.1f}%")
        print(f"  - Confidence: {result['confidence_level']}")
        print(f"  - Strong factors: {len(insights.get('strong_factors', []))} identified")
        print(f"  - Weak factors: {len(insights.get('weak_factors', []))} identified")
        print(f"  - Recommendations: {len(insights.get('recommendations', []))} provided")
    except Exception as e:
        print(f"✗ Error in prediction pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Model metrics
    print("\n[TEST 5] Checking model metrics...")
    try:
        metrics = manager.metrics
        if not metrics:
            print("✗ No metrics available")
            return False
        
        print("✓ Model metrics available")
        print(f"  - Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"  - Precision: {metrics.get('precision', 0):.4f}")
        print(f"  - Recall: {metrics.get('recall', 0):.4f}")
        print(f"  - F1-Score: {metrics.get('f1', 0):.4f}")
        print(f"  - AUC-ROC: {metrics.get('auc_roc', 0):.4f}")
    except Exception as e:
        print(f"✗ Error reading metrics: {e}")
        return False
    
    # Test 6: Feature importance
    print("\n[TEST 6] Checking feature importance...")
    try:
        importance = manager.feature_importance
        if not importance:
            print("✗ No feature importance available")
            return False
        
        print("✓ Feature importance available")
        for i, (feature, imp) in enumerate(list(importance.items())[:3], 1):
            print(f"  {i}. {feature}: {imp:.4f}")
    except Exception as e:
        print(f"✗ Error reading feature importance: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED - APPLICATION IS FULLY FUNCTIONAL!")
    print("=" * 70)
    print("\n📝 To start the Streamlit application, run:")
    print("   streamlit run app/main.py")
    print("\n🎓 The application includes:")
    print("   • Student Placement Prediction with AI insights")
    print("   • Student Registration and AI Interview")
    print("   • Analytics and Model Performance Dashboard")
    print("   • Interview Feedback and Improvement Suggestions")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
