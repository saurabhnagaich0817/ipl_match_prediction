"""
Configuration file for IPL Match Prediction Project
"""

import os

# Data Configuration
DATA_CONFIG = {
    'dataset_path': 'ipl_matches.csv',
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'create_historical_features': True,
    'create_venue_features': True, 
    'create_toss_features': True,
    'create_season_features': True,
    'scale_features': True,
    'min_matches_for_stats': 5  # Minimum matches for reliable statistics
}

# Model Configuration
MODEL_CONFIG = {
    'default_model': 'Logistic_Regression',
    'models_to_evaluate': ['Logistic_Regression', 'Random_Forest', 'Gradient_Boosting'],
    'hyperparameter_tuning': True,
    'class_weight': 'balanced'
}

# Hyperparameter Grids
HYPERPARAMETER_GRIDS = {
    'Logistic_Regression': {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000]
    },
    'Random_Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced']
    },
    'Gradient_Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }
}

# Quick hyperparameter grids for faster execution
QUICK_HYPERPARAMETER_GRIDS = {
    'Logistic_Regression': {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['liblinear']
    },
    'Random_Forest': {
        'n_estimators': [50, 100],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    },
    'Gradient_Boosting': {
        'n_estimators': [50, 100],
        'learning_rate': [0.1, 0.2],
        'max_depth': [3, 5]
    }
}

# Team Name Mappings
TEAM_NAME_MAPPINGS = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Rising Pune Supergiants': 'Rising Pune Supergiant',
    'Rising Pune Supergiants': 'Rising Pune Supergiant',
    'Pune Warriors': 'Pune Warriors India',
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Gujarat Lions': 'Gujarat Lions'
}

# Column Name Mappings (in case dataset has different column names)
COLUMN_MAPPINGS = {
    'team1': ['team1', 'Team1', 'Team 1', 'home_team'],
    'team2': ['team2', 'Team2', 'Team 2', 'away_team'],
    'winner': ['winner', 'Winner', 'match_winner', 'result'],
    'venue': ['venue', 'Venue', 'ground', 'stadium'],
    'city': ['city', 'City', 'location'],
    'toss_winner': ['toss_winner', 'Toss_winner', 'toss_winner'],
    'toss_decision': ['toss_decision', 'Toss_decision', 'toss_choice'],
    'date': ['date', 'Date', 'match_date', 'Date_played']
}

# Output Configuration
OUTPUT_CONFIG = {
    'save_model': True,
    'model_filename': 'ipl_prediction_model.pkl',
    'save_plots': True,
    'plot_format': 'png',
    'plot_dpi': 300,
    'results_directory': 'results',
    'models_directory': 'models'
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_format': '%(asctime)s - %(levelname)s - %(message)s',
    'log_file': 'ipl_prediction.log'
}

# Evaluation Metrics
EVALUATION_METRICS = [
    'accuracy',
    'precision', 
    'recall',
    'f1',
    'roc_auc'
]

# Plotting Configuration
PLOT_CONFIG = {
    'figsize': (12, 8),
    'style': 'seaborn-v0_8',
    'color_palette': 'viridis',
    'save_plots': True
}

# Expected Dataset Columns
REQUIRED_COLUMNS = ['team1', 'team2', 'winner']
OPTIONAL_COLUMNS = ['venue', 'city', 'toss_winner', 'toss_decision', 'date', 'id']

# Feature Categories
FEATURE_CATEGORIES = {
    'basic_features': ['team1_encoded', 'team2_encoded', 'venue_encoded'],
    'toss_features': ['toss_winner_encoded', 'toss_decision_encoded', 'toss_advantage', 'bat_first'],
    'historical_features': ['team1_win_rate', 'team2_win_rate', 'win_rate_diff', 'head2head_team1'],
    'venue_features': ['team1_venue_win_rate', 'team2_venue_win_rate'],
    'temporal_features': ['tournament_stage_encoded', 'match_number', 'season']
}

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        OUTPUT_CONFIG['results_directory'],
        OUTPUT_CONFIG['models_directory']
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def get_hyperparameter_grid(model_name, quick=True):
    """Get hyperparameter grid for a model"""
    grids = QUICK_HYPERPARAMETER_GRIDS if quick else HYPERPARAMETER_GRIDS
    return grids.get(model_name, {})

def validate_dataset_columns(columns):
    """Validate if dataset has required columns"""
    missing_required = [col for col in REQUIRED_COLUMNS if col not in columns]
    
    if missing_required:
        print(f"Missing required columns: {missing_required}")
        return False
    
    available_optional = [col for col in OPTIONAL_COLUMNS if col in columns]
    print(f"Available optional columns: {available_optional}")
    
    return True