"""
Model Training Script
Run this to train the ML model
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocess import DataPreprocessor
from models.train import ModelPipeline


def main():
    """Main training function."""
    
    print("=" * 60)
    print("STUDENT PLACEMENT PREDICTION - MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading Data...")
    preprocessor = DataPreprocessor("data/placement_data.csv")
    df = preprocessor.load_data()
    
    # Validate data
    print("[2/4] Validating Data...")
    preprocessor.validate_data()
    
    # Get data summary
    summary = preprocessor.get_data_summary()
    print(f"  • Total Records: {summary['total_records']}")
    print(f"  • Placed: {summary['placed_count']} ({summary['placement_rate']:.1f}%)")
    print(f"  • Not Placed: {summary['not_placed_count']}")
    print(f"  • Features: {len(summary['features'])}")
    
    # Extract features
    print("\n[3/4] Extracting Features...")
    X, y = preprocessor.get_features_and_target()
    print(f"  • Features Shape: {X.shape}")
    print(f"  • Target Shape: {y.shape}")
    
    # Train model
    print("\n[4/4] Training Model...")
    pipeline = ModelPipeline(model_dir="models")
    results = pipeline.train(X, y)
    
    # Display results
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"  • Train Accuracy:  {results['train_accuracy']:.4f}")
    print(f"  • Test Accuracy:   {results['test_accuracy']:.4f}")
    print(f"  • Precision:       {results['precision']:.4f}")
    print(f"  • Recall:          {results['recall']:.4f}")
    print(f"  • F1-Score:        {results['f1']:.4f}")
    print(f"  • AUC-ROC:         {results['auc_roc']:.4f}")
    
    # Save model
    print("\n[SAVING] Saving Model and Scaler...")
    model_path, scaler_path = pipeline.save_model()
    print(f"  • Model saved to: {model_path}")
    print(f"  • Scaler saved to: {scaler_path}")
    
    # Feature importance
    print("\n[FEATURE IMPORTANCE] Top 5 Features:")
    importance = pipeline.get_feature_importance()
    for i, (feature, imp) in enumerate(list(importance.items())[:5], 1):
        print(f"  {i}. {feature:20s}: {imp:.4f}")
    
    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nYou can now run: streamlit run app/main.py")


if __name__ == "__main__":
    main()
