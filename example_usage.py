"""
Example usage of the IPL Match Prediction system
This file demonstrates different ways to use the project
"""

import pandas as pd
import numpy as np
from data_pipeline import IPLDataPipeline
from feature_engineering import IPLFeatureEngineer
from model_training import IPLModelTrainer
from utils import *
import matplotlib.pyplot as plt

def example_basic_usage():
    """
    Example 1: Basic usage of the prediction system
    """
    print("="*60)
    print("EXAMPLE 1: BASIC USAGE")
    print("="*60)
    
    # Load and process data
    pipeline = IPLDataPipeline("ipl_matches.csv")
    data = pipeline.load_data()
    
    if data is None:
        print("Please ensure dataset is available")
        return
    
    # Basic processing
    data = pipeline.clean_data()
    data = pipeline.encode_categorical_variables()
    
    # Prepare features and target
    X, y = pipeline.prepare_features_target()
    
    # Split data
    X_train, X_test, y_train, y_test = pipeline.split_data(X, y)
    
    # Train model
    trainer = IPLModelTrainer()
    X_train_scaled, y_train = trainer.prepare_data(X_train, y_train)
    best_model = trainer.train_best_model(X_train_scaled, y_train)
    
    print("Basic usage completed successfully!")

def example_advanced_feature_engineering():
    """
    Example 2: Advanced feature engineering
    """
    print("="*60)
    print("EXAMPLE 2: ADVANCED FEATURE ENGINEERING")
    print("="*60)
    
    # Load data
    pipeline = IPLDataPipeline("ipl_matches.csv")
    data = pipeline.load_data()
    
    if data is None:
        return
    
    # Clean and encode
    data = pipeline.clean_data()
    data = pipeline.create_team_mapping()
    data = pipeline.encode_categorical_variables()
    
    # Advanced feature engineering
    feature_engineer = IPLFeatureEngineer(data)
    
    # Create different types of features
    data = feature_engineer.create_historical_features()
    data = feature_engineer.create_win_rate_features()
    data = feature_engineer.create_venue_features()
    data = feature_engineer.create_toss_features()
    
    print("Features created:")
    print(data.columns.tolist())
    
    print("Advanced feature engineering completed!")

def example_model_comparison():
    """
    Example 3: Compare multiple models
    """
    print("="*60)
    print("EXAMPLE 3: MODEL COMPARISON")
    print("="*60)
    
    # Load and prepare data (shortened version)
    pipeline = IPLDataPipeline("ipl_matches.csv")
    data = pipeline.load_data()
    
    if data is None:
        return
    
    # Quick processing
    data = pipeline.clean_data()
    data = pipeline.encode_categorical_variables()
    X, y = pipeline.prepare_features_target()
    
    # Initialize trainer
    trainer = IPLModelTrainer()
    trainer.initialize_models()
    
    # Compare all models
    results = trainer.evaluate_models_cv(X, y)
    
    # Plot comparison
    plot_model_comparison(results, metric='accuracy', save_plot=True)
    
    print("Model comparison completed!")

def example_custom_prediction():
    """
    Example 4: Make custom predictions
    """
    print("="*60)
    print("EXAMPLE 4: CUSTOM PREDICTIONS")
    print("="*60)
    
    # Load trained model
    trainer = IPLModelTrainer()
    
    if trainer.load_model('ipl_prediction_model.pkl'):
        # Make predictions for hypothetical matches
        matches = [
            {
                'team1': 'Mumbai Indians',
                'team2': 'Chennai Super Kings',
                'venue': 'Wankhede Stadium',
                'toss_winner': 'Mumbai Indians',
                'toss_decision': 'bat'
            },
            {
                'team1': 'Royal Challengers Bangalore',
                'team2': 'Delhi Capitals',
                'venue': 'M. Chinnaswamy Stadium',
                'toss_winner': 'Delhi Capitals',
                'toss_decision': 'field'
            }
        ]
        
        for i, match in enumerate(matches, 1):
            print(f"\nPrediction {i}:")
            prediction = trainer.predict_match(**match)
    else:
        print("No trained model found. Please run the main pipeline first.")

def example_data_analysis():
    """
    Example 5: Data analysis and visualization
    """
    print("="*60)
    print("EXAMPLE 5: DATA ANALYSIS")
    print("="*60)
    
    # Load data
    data = load_and_validate_dataset("ipl_matches.csv")
    
    if data is None:
        return
    
    # Basic analysis
    print("\nDataset Overview:")
    print(f"Total matches: {len(data)}")
    print(f"Date range: {data['date'].min() if 'date' in data.columns else 'N/A'} to {data['date'].max() if 'date' in data.columns else 'N/A'}")
    
    # Team analysis
    if 'team1' in data.columns and 'team2' in data.columns and 'winner' in data.columns:
        team_performance = analyze_team_performance(data, {})
        print("\nTop 5 Teams by Win Rate:")
        print(team_performance.head())
        
        # Venue analysis
        if 'venue' in data.columns:
            venue_performance = analyze_venue_performance(data)
            print("\nTop 5 Venues by Matches Played:")
            print(venue_performance.head())
        
        # Create timeline
        create_match_timeline(data, save_plot=True)

def example_feature_importance_analysis():
    """
    Example 6: Analyze feature importance
    """
    print("="*60)
    print("EXAMPLE 6: FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Load trained model
    trainer = IPLModelTrainer()
    
    if trainer.load_model('ipl_prediction_model.pkl'):
        # Get feature importance
        if hasattr(trainer.best_model, 'feature_importances_'):
            importance_df = create_feature_importance_table(
                trainer.best_model, 
                trainer.feature_names,
                top_n=10
            )
            
            print("Top 10 Most Important Features:")
            print(importance_df)
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            plt.barh(importance_df['feature'][::-1], importance_df['importance'][::-1])
            plt.title('Top 10 Feature Importances')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print("Model doesn't have feature importances")
    else:
        print("No trained model found")

def example_prediction_analysis():
    """
    Example 7: Analyze predictions in detail
    """
    print("="*60)
    print("EXAMPLE 7: PREDICTION ANALYSIS")
    print("="*60)
    
    # This would typically use actual test data
    # For demonstration, we'll create synthetic data
    
    np.random.seed(42)
    y_true = np.random.choice([0, 1], size=100)
    y_prob = np.random.random(100)
    y_pred = (y_prob > 0.5).astype(int)
    
    # Generate prediction report
    report = generate_prediction_report(y_true, y_pred, y_prob)
    
    print("Prediction Report:")
    print(f"Accuracy: {report['accuracy']:.3f}")
    print(f"Total Predictions: {report['total_predictions']}")
    print(f"Correct Predictions: {report['correct_predictions']}")
    
    # Plot prediction distribution
    plot_prediction_distribution(y_prob, save_plot=True)
    
    # Validation
    validation = validate_predictions(y_true, y_pred)
    print(f"Validation Accuracy: {validation['overall_accuracy']:.3f}")

def run_all_examples():
    """
    Run all examples sequentially
    """
    examples = [
        example_basic_usage,
        example_advanced_feature_engineering,
        example_model_comparison,
        example_custom_prediction,
        example_data_analysis,
        example_feature_importance_analysis,
        example_prediction_analysis
    ]
    
    print("RUNNING ALL EXAMPLES")
    print("="*60)
    
    for i, example_func in enumerate(examples, 1):
        try:
            print(f"\n\nRunning Example {i}...")
            example_func()
            print(f"Example {i} completed successfully!")
        except Exception as e:
            print(f"Example {i} failed with error: {str(e)}")
            continue
    
    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETED")
    print("="*60)

if __name__ == "__main__":
    # Print system info
    print_system_info()
    
    # Create necessary directories
    from config import create_directories
    create_directories()
    
    # Run examples
    print("\nSelect an example to run:")
    print("1. Basic Usage")
    print("2. Advanced Feature Engineering")
    print("3. Model Comparison")
    print("4. Custom Predictions")
    print("5. Data Analysis")
    print("6. Feature Importance Analysis")
    print("7. Prediction Analysis")
    print("8. Run All Examples")
    
    choice = input("\nEnter your choice (1-8): ").strip()
    
    examples = {
        '1': example_basic_usage,
        '2': example_advanced_feature_engineering,
        '3': example_model_comparison,
        '4': example_custom_prediction,
        '5': example_data_analysis,
        '6': example_feature_importance_analysis,
        '7': example_prediction_analysis,
        '8': run_all_examples
    }
    
    if choice in examples:
        examples[choice]()
    else:
        print("Invalid choice. Running basic usage example...")
        example_basic_usage()