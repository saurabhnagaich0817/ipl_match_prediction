"""
Utility functions for IPL Match Prediction Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_validate_dataset(filepath, required_columns=None):
    """
    Load and validate dataset
    """
    if required_columns is None:
        required_columns = ['team1', 'team2', 'winner']
    
    try:
        data = pd.read_csv(filepath)
        print(f"Dataset loaded successfully: {data.shape}")
        
        # Check for required columns
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            print(f"Warning: Missing required columns: {missing_cols}")
            return None
        
        print(f"Available columns: {data.columns.tolist()}")
        return data
        
    except FileNotFoundError:
        print(f"Error: Dataset file '{filepath}' not found")
        print("Please ensure you have downloaded an IPL dataset and placed it in the project directory")
        return None
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

def create_results_summary(cv_results, final_results, model_name):
    """
    Create a comprehensive results summary
    """
    summary = {
        'model_name': model_name,
        'cv_accuracy_mean': cv_results['accuracy'].mean(),
        'cv_accuracy_std': cv_results['accuracy'].std(),
        'cv_precision_mean': cv_results['precision'].mean(),
        'cv_recall_mean': cv_results['recall'].mean(),
        'cv_f1_mean': cv_results['f1'].mean(),
        'cv_roc_auc_mean': cv_results['roc_auc'].mean(),
        'test_roc_auc': final_results['test_score'],
        'train_roc_auc': final_results['train_score'],
        'overfitting_score': final_results['train_score'] - final_results['test_score']
    }
    
    return summary

def plot_model_comparison(results_dict, metric='roc_auc', save_plot=False):
    """
    Plot comparison of different models
    """
    models = list(results_dict.keys())
    scores = [results_dict[model][metric].mean() for model in models]
    errors = [results_dict[model][metric].std() for model in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, scores, yerr=errors, capsize=5, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    plt.title(f'Model Comparison - {metric.replace("_", " ").title()}')
    plt.ylabel(metric.replace("_", " ").title())
    plt.xlabel('Models')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, score, error in zip(bars, scores, errors):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + error,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(f'model_comparison_{metric}.png', dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_learning_curve(model, X, y, cv=5, save_plot=False):
    """
    Plot learning curve for a model
    """
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    
    plt.plot(train_sizes, val_mean, 'o-', label='Cross-Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    
    plt.title('Learning Curve')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
    
    plt.show()

def generate_prediction_report(y_true, y_pred, y_prob, team_names=None):
    """
    Generate a comprehensive prediction report
    """
    report = {
        'classification_report': classification_report(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'accuracy': (y_true == y_pred).mean(),
        'total_predictions': len(y_true),
        'correct_predictions': (y_true == y_pred).sum()
    }
    
    if team_names:
        # Add team-specific analysis
        team1_wins_actual = y_true.sum()
        team1_wins_predicted = y_pred.sum()
        
        report['team1_wins_actual'] = team1_wins_actual
        report['team1_wins_predicted'] = team1_wins_predicted
        report['team2_wins_actual'] = len(y_true) - team1_wins_actual
        report['team2_wins_predicted'] = len(y_pred) - team1_wins_predicted
    
    return report

def save_results_to_csv(results_dict, filename='model_results.csv'):
    """
    Save model results to CSV file
    """
    results_df = pd.DataFrame(results_dict).T
    results_df.to_csv(filename)
    print(f"Results saved to {filename}")

def create_feature_importance_table(model, feature_names, top_n=15):
    """
    Create a table of feature importances
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print("Model doesn't have feature importances or coefficients")
        return None
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return feature_importance_df.head(top_n)

def plot_prediction_distribution(y_prob, save_plot=False):
    """
    Plot distribution of prediction probabilities
    """
    plt.figure(figsize=(10, 6))
    plt.hist(y_prob, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
    plt.title('Distribution of Prediction Probabilities')
    plt.xlabel('Predicted Probability (Team 1 Win)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('prediction_distribution.png', dpi=300, bbox_inches='tight')
    
    plt.show()

def analyze_team_performance(data, results):
    """
    Analyze performance by team
    """
    team_stats = {}
    
    # Get unique teams
    teams = set(data['team1'].unique()) | set(data['team2'].unique())
    
    for team in teams:
        team_matches = data[(data['team1'] == team) | (data['team2'] == team)]
        team_wins = len(team_matches[team_matches['winner'] == team])
        total_matches = len(team_matches)
        win_rate = team_wins / total_matches if total_matches > 0 else 0
        
        team_stats[team] = {
            'total_matches': total_matches,
            'wins': team_wins,
            'losses': total_matches - team_wins,
            'win_rate': win_rate
        }
    
    return pd.DataFrame(team_stats).T.sort_values('win_rate', ascending=False)

def analyze_venue_performance(data):
    """
    Analyze performance by venue
    """
    venue_stats = {}
    
    for venue in data['venue'].unique():
        if pd.isna(venue):
            continue
            
        venue_matches = data[data['venue'] == venue]
        total_matches = len(venue_matches)
        
        if total_matches == 0:
            continue
            
        # Calculate home advantage (assuming team1 is home team more often)
        team1_wins = len(venue_matches[venue_matches['winner'] == venue_matches['team1']])
        home_advantage = team1_wins / total_matches
        
        venue_stats[venue] = {
            'total_matches': total_matches,
            'team1_wins': team1_wins,
            'team2_wins': total_matches - team1_wins,
            'home_advantage': home_advantage
        }
    
    return pd.DataFrame(venue_stats).T.sort_values('total_matches', ascending=False)

def create_match_timeline(data, save_plot=False):
    """
    Create a timeline of matches if date information is available
    """
    if 'date' not in data.columns:
        print("Date column not available for timeline analysis")
        return None
    
    try:
        data['date'] = pd.to_datetime(data['date'])
        data['year'] = data['date'].dt.year
        
        matches_per_year = data['year'].value_counts().sort_index()
        
        plt.figure(figsize=(12, 6))
        matches_per_year.plot(kind='bar')
        plt.title('IPL Matches per Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Matches')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('matches_timeline.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return matches_per_year
        
    except Exception as e:
        print(f"Error creating timeline: {str(e)}")
        return None

def validate_predictions(y_true, y_pred, confidence_threshold=0.7):
    """
    Validate predictions and identify high-confidence predictions
    """
    # Basic validation metrics
    accuracy = (y_true == y_pred).mean()
    
    # High confidence predictions (for probability-based models)
    validation_results = {
        'overall_accuracy': accuracy,
        'total_predictions': len(y_true),
        'correct_predictions': (y_true == y_pred).sum(),
        'incorrect_predictions': (y_true != y_pred).sum()
    }
    
    return validation_results

def create_project_summary():
    """
    Create a project summary with key information
    """
    summary = {
        'project_name': 'IPL Match Outcome Prediction',
        'created_date': datetime.now().strftime('%Y-%m-%d'),
        'description': 'Machine Learning project to predict IPL match outcomes',
        'algorithms_used': ['Logistic Regression', 'Random Forest', 'Gradient Boosting'],
        'key_features': [
            'Historical team performance',
            'Venue-specific statistics', 
            'Toss impact analysis',
            'Head-to-head records',
            'Tournament stage analysis'
        ],
        'evaluation_metrics': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        'files_created': [
            'data_pipeline.py',
            'feature_engineering.py', 
            'model_training.py',
            'main.py',
            'config.py',
            'utils.py',
            'requirements.txt',
            'README.md'
        ]
    }
    
    return summary

def print_system_info():
    """
    Print system and library information
    """
    import sys
    import sklearn
    import pandas as pd
    import numpy as np
    
    print("="*50)
    print("SYSTEM INFORMATION")
    print("="*50)
    print(f"Python Version: {sys.version}")
    print(f"Pandas Version: {pd.__version__}")
    print(f"NumPy Version: {np.__version__}")
    print(f"Scikit-learn Version: {sklearn.__version__}")
    print("="*50)

def cleanup_temp_files():
    """
    Clean up temporary files created during execution
    """
    temp_files = [
        'ipl_prediction.log',
        '.DS_Store'
    ]
    
    for file in temp_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Cleaned up: {file}")

def export_model_info(model, feature_names, filename='model_info.txt'):
    """
    Export model information to a text file
    """
    with open(filename, 'w') as f:
        f.write("IPL MATCH PREDICTION MODEL INFORMATION\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model Type: {type(model).__name__}\n")
        f.write(f"Number of Features: {len(feature_names)}\n")
        f.write(f"Feature Names: {', '.join(feature_names)}\n\n")
        
        if hasattr(model, 'get_params'):
            f.write("Model Parameters:\n")
            for param, value in model.get_params().items():
                f.write(f"  {param}: {value}\n")
        
        f.write(f"\nModel saved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Model information exported to {filename}")

# Error handling utilities
class IPLPredictionError(Exception):
    """Custom exception for IPL prediction errors"""
    pass

def handle_missing_data(data, strategy='drop'):
    """
    Handle missing data in the dataset
    """
    if strategy == 'drop':
        initial_shape = data.shape
        data = data.dropna()
        print(f"Dropped rows with missing data: {initial_shape[0] - data.shape[0]} rows removed")
    elif strategy == 'fill':
        # Fill with appropriate values
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col].fillna('Unknown', inplace=True)
            else:
                data[col].fillna(data[col].median(), inplace=True)
        print("Missing values filled with appropriate defaults")
    
    return data