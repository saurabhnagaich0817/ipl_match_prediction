"""
IPL Match Outcome Prediction
Main script to run the complete pipeline
"""

import pandas as pd
import numpy as np
from data_pipeline import IPLDataPipeline
from feature_engineering import IPLFeatureEngineer
from model_training import IPLModelTrainer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def main():
    """
    Main function to run the complete IPL prediction pipeline
    """
    print("="*60)
    print("IPL MATCH OUTCOME PREDICTION PIPELINE")
    print("="*60)
    
    # Configuration
    DATA_PATH = "ipl_matches.csv"  # Update this with your dataset path
    MODEL_NAME = "Logistic_Regression"  # Choose: Logistic_Regression, Random_Forest, Gradient_Boosting
    
    try:
        # Step 1: Data Pipeline
        print("\nStep 1: Loading and Processing Data...")
        pipeline = IPLDataPipeline(DATA_PATH)
        
        # Load data
        data = pipeline.load_data()
        if data is None:
            print("Please ensure the dataset is available and update the DATA_PATH")
            return
        
        # Explore data
        pipeline.explore_data()
        
        # Clean and process data
        data = pipeline.clean_data()
        data = pipeline.create_team_mapping()
        data = pipeline.encode_categorical_variables()
        
        # Step 2: Feature Engineering
        print("\nStep 2: Feature Engineering...")
        feature_engineer = IPLFeatureEngineer(data)
        engineered_data, feature_columns = feature_engineer.run_feature_engineering()
        
        # Step 3: Prepare features and target
        print("\nStep 3: Preparing Features and Target...")
        X, y = pipeline.prepare_features_target()
        
        if X is None or y is None:
            print("Error in preparing features and target")
            return
        
        # Add engineered features
        engineered_features = [col for col in feature_columns if col in engineered_data.columns]
        if engineered_features:
            X_engineered = engineered_data[engineered_features]
            # Align indices
            X_engineered = X_engineered.loc[X.index]
            X = pd.concat([X, X_engineered], axis=1)
        
        print(f"Final feature set shape: {X.shape}")
        print(f"Features: {X.columns.tolist()}")
        
        # Step 4: Split data
        print("\nStep 4: Splitting Data...")
        X_train, X_test, y_train, y_test = pipeline.split_data(X, y, test_size=0.2)
        
        # Step 5: Model Training and Evaluation
        print("\nStep 5: Model Training and Evaluation...")
        trainer = IPLModelTrainer()
        
        # Prepare data for modeling
        X_train_scaled, y_train = trainer.prepare_data(X_train, y_train)
        X_test_scaled = trainer.scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        # Initialize models
        trainer.initialize_models()
        
        # Cross-validation evaluation
        cv_results = trainer.evaluate_models_cv(X_train_scaled, y_train)
        
        # Train best model
        best_model = trainer.train_best_model(X_train_scaled, y_train, MODEL_NAME)
        
        # Final evaluation
        final_results = trainer.evaluate_final_model(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        
        # Step 6: Save Model
        print("\nStep 6: Saving Model...")
        trainer.save_model('ipl_prediction_model.pkl')
        
        # Step 7: Model Summary
        trainer.model_summary()
        
        # Step 8: Sample Predictions (if you want to test)
        print("\nStep 8: Sample Prediction...")
        print("Note: This is a simplified prediction. In practice, you would need")
        print("to properly encode team names and calculate all features.")
        
        sample_prediction = trainer.predict_match(
            team1="Mumbai Indians",
            team2="Chennai Super Kings",
            venue="Wankhede Stadium",
            toss_winner="Mumbai Indians",
            toss_decision="bat"
        )
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return {
            'model': trainer,
            'results': final_results,
            'cv_results': cv_results
        }
        
    except Exception as e:
        print(f"Error in pipeline execution: {str(e)}")
        print("Please check your dataset format and ensure all required columns are present")
        return None

def analyze_results(results):
    """
    Analyze and display detailed results
    """
    if results is None:
        return
    
    print("\n" + "="*50)
    print("DETAILED RESULTS ANALYSIS")
    print("="*50)
    
    cv_results = results['cv_results']
    final_results = results['results']
    
    # Cross-validation results summary
    print("\nCross-Validation Results Summary:")
    for model_name, scores in cv_results.items():
        print(f"\n{model_name}:")
        for metric, values in scores.items():
            print(f"  {metric.capitalize()}: {values.mean():.4f} ± {values.std():.4f}")
    
    # Final model performance
    print(f"\nFinal Model Test Performance:")
    print(f"  ROC-AUC Score: {final_results['test_score']:.4f}")
    print(f"  Training Score: {final_results['train_score']:.4f}")
    
    # Check for overfitting
    score_diff = final_results['train_score'] - final_results['test_score']
    if score_diff > 0.1:
        print(f"  ⚠️  Potential overfitting detected (difference: {score_diff:.4f})")
    else:
        print(f"  ✅ Good generalization (difference: {score_diff:.4f})")

if __name__ == "__main__":
    # Run the main pipeline
    results = main()
    
    # Analyze results
    if results:
        analyze_results(results)