import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class IPLDataPipeline:
    def __init__(self, data_path):
        """
        Initialize the data pipeline with dataset path
        """
        self.data_path = data_path
        self.data = None
        self.label_encoders = {}
        
    def load_data(self):
        """
        Load the IPL dataset
        """
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Dataset loaded successfully with shape: {self.data.shape}")
            print("\nColumns in dataset:")
            print(self.data.columns.tolist())
            return self.data
        except FileNotFoundError:
            print(f"Error: Dataset not found at {self.data_path}")
            return None
    
    def explore_data(self):
        """
        Basic data exploration
        """
        if self.data is None:
            print("Please load data first using load_data()")
            return
        
        print("="*50)
        print("DATA EXPLORATION")
        print("="*50)
        
        print(f"\nDataset shape: {self.data.shape}")
        print(f"\nMissing values:")
        print(self.data.isnull().sum())
        
        print(f"\nData types:")
        print(self.data.dtypes)
        
        print(f"\nFirst few rows:")
        print(self.data.head())
        
        # Basic statistics for numerical columns
        print(f"\nNumerical columns statistics:")
        print(self.data.describe())
        
    def clean_data(self):
        """
        Clean the dataset by handling missing values and inconsistencies
        """
        if self.data is None:
            print("Please load data first")
            return
        
        print("Cleaning data...")
        initial_shape = self.data.shape
        
        # Remove rows with missing critical values
        critical_columns = ['team1', 'team2', 'winner']
        self.data = self.data.dropna(subset=critical_columns)
        
        # Handle missing values in other columns
        if 'city' in self.data.columns:
            self.data['city'].fillna('Unknown', inplace=True)
        
        if 'venue' in self.data.columns:
            self.data['venue'].fillna('Unknown', inplace=True)
            
        if 'toss_winner' in self.data.columns:
            self.data['toss_winner'].fillna('Unknown', inplace=True)
            
        if 'toss_decision' in self.data.columns:
            self.data['toss_decision'].fillna('bat', inplace=True)
        
        # Remove matches with no result or tie (for binary classification)
        if 'result' in self.data.columns:
            self.data = self.data[self.data['result'] == 'normal']
        
        # Remove duplicate matches
        if 'id' in self.data.columns:
            self.data = self.data.drop_duplicates(subset=['id'])
        else:
            self.data = self.data.drop_duplicates()
        
        print(f"Data cleaned. Shape changed from {initial_shape} to {self.data.shape}")
        return self.data
    
    def create_team_mapping(self):
        """
        Create consistent team name mapping
        """
        team_mapping = {
            'Delhi Daredevils': 'Delhi Capitals',
            'Kings XI Punjab': 'Punjab Kings',
            'Rising Pune Supergiants': 'Rising Pune Supergiant',
            'Pune Warriors': 'Pune Warriors India'
        }
        
        for col in ['team1', 'team2', 'winner', 'toss_winner']:
            if col in self.data.columns:
                self.data[col] = self.data[col].replace(team_mapping)
        
        return self.data
    
    def encode_categorical_variables(self):
        """
        Encode categorical variables using label encoding
        """
        if self.data is None:
            print("Please load and clean data first")
            return
        
        categorical_columns = ['team1', 'team2', 'city', 'venue', 'toss_winner', 'toss_decision']
        
        for col in categorical_columns:
            if col in self.data.columns:
                le = LabelEncoder()
                self.data[f'{col}_encoded'] = le.fit_transform(self.data[col].astype(str))
                self.label_encoders[col] = le
        
        print("Categorical variables encoded successfully")
        return self.data
    
    def prepare_features_target(self, target_column='winner'):
        """
        Prepare features and target variable for modeling
        """
        if self.data is None:
            print("Please process data first")
            return None, None
        
        # Feature columns (encoded versions)
        feature_columns = [
            'team1_encoded', 'team2_encoded', 'city_encoded', 
            'venue_encoded', 'toss_winner_encoded', 'toss_decision_encoded'
        ]
        
        # Filter only available columns
        available_features = [col for col in feature_columns if col in self.data.columns]
        
        if len(available_features) == 0:
            print("No encoded features available. Please run encode_categorical_variables() first")
            return None, None
        
        X = self.data[available_features]
        
        # Create binary target (1 if team1 wins, 0 if team2 wins)
        y = (self.data[target_column] == self.data['team1']).astype(int)
        
        print(f"Features prepared with shape: {X.shape}")
        print(f"Target variable prepared with shape: {y.shape}")
        print(f"Target distribution: {y.value_counts()}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Data split completed:")
        print(f"Training set: {X_train.shape}")
        print(f"Testing set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def get_processed_data(self):
        """
        Get the processed dataset
        """
        return self.data
    
    def get_label_encoders(self):
        """
        Get the fitted label encoders
        """
        return self.label_encoders