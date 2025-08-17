import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import defaultdict

class IPLFeatureEngineer:
    def __init__(self, data):
        """
        Initialize feature engineering with processed data
        """
        self.data = data.copy()
        self.team_stats = defaultdict(dict)
        self.venue_stats = defaultdict(dict)
        self.scaler = StandardScaler()
        
    def create_historical_features(self):
        """
        Create historical performance features for teams
        """
        print("Creating historical features...")
        
        # Sort by date if available
        if 'date' in self.data.columns:
            self.data = self.data.sort_values('date')
        elif 'id' in self.data.columns:
            self.data = self.data.sort_values('id')
        
        # Initialize feature columns
        self.data['team1_wins'] = 0
        self.data['team2_wins'] = 0
        self.data['team1_matches'] = 0
        self.data['team2_matches'] = 0
        self.data['head2head_team1'] = 0
        
        team_stats = defaultdict(lambda: {'wins': 0, 'matches': 0})
        head_to_head = defaultdict(lambda: defaultdict(int))
        
        for idx, row in self.data.iterrows():
            team1, team2 = row['team1'], row['team2']
            winner = row['winner']
            
            # Historical stats before this match
            self.data.at[idx, 'team1_wins'] = team_stats[team1]['wins']
            self.data.at[idx, 'team2_wins'] = team_stats[team2]['wins']
            self.data.at[idx, 'team1_matches'] = team_stats[team1]['matches']
            self.data.at[idx, 'team2_matches'] = team_stats[team2]['matches']
            self.data.at[idx, 'head2head_team1'] = head_to_head[team1][team2]
            
            # Update stats after this match
            team_stats[team1]['matches'] += 1
            team_stats[team2]['matches'] += 1
            
            if winner == team1:
                team_stats[team1]['wins'] += 1
                head_to_head[team1][team2] += 1
            elif winner == team2:
                team_stats[team2]['wins'] += 1
                head_to_head[team2][team1] += 1
        
        print("Historical features created successfully")
        return self.data
    
    def create_win_rate_features(self):
        """
        Create win rate features
        """
        print("Creating win rate features...")
        
        # Calculate win rates
        self.data['team1_win_rate'] = np.where(
            self.data['team1_matches'] > 0,
            self.data['team1_wins'] / self.data['team1_matches'],
            0.5  # Default win rate for new teams
        )
        
        self.data['team2_win_rate'] = np.where(
            self.data['team2_matches'] > 0,
            self.data['team2_wins'] / self.data['team2_matches'],
            0.5  # Default win rate for new teams
        )
        
        # Win rate difference
        self.data['win_rate_diff'] = self.data['team1_win_rate'] - self.data['team2_win_rate']
        
        print("Win rate features created successfully")
        return self.data
    
    def create_venue_features(self):
        """
        Create venue-based features
        """
        print("Creating venue features...")
        
        if 'venue' not in self.data.columns:
            print("Venue column not found, skipping venue features")
            return self.data
        
        # Initialize venue features
        self.data['team1_venue_wins'] = 0
        self.data['team1_venue_matches'] = 0
        self.data['team2_venue_wins'] = 0
        self.data['team2_venue_matches'] = 0
        
        venue_team_stats = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'matches': 0}))
        
        for idx, row in self.data.iterrows():
            team1, team2 = row['team1'], row['team2']
            venue = row['venue']
            winner = row['winner']
            
            # Historical venue stats before this match
            self.data.at[idx, 'team1_venue_wins'] = venue_team_stats[venue][team1]['wins']
            self.data.at[idx, 'team1_venue_matches'] = venue_team_stats[venue][team1]['matches']
            self.data.at[idx, 'team2_venue_wins'] = venue_team_stats[venue][team2]['wins']
            self.data.at[idx, 'team2_venue_matches'] = venue_team_stats[venue][team2]['matches']
            
            # Update venue stats after this match
            venue_team_stats[venue][team1]['matches'] += 1
            venue_team_stats[venue][team2]['matches'] += 1
            
            if winner == team1:
                venue_team_stats[venue][team1]['wins'] += 1
            elif winner == team2:
                venue_team_stats[venue][team2]['wins'] += 1
        
        # Calculate venue win rates
        self.data['team1_venue_win_rate'] = np.where(
            self.data['team1_venue_matches'] > 0,
            self.data['team1_venue_wins'] / self.data['team1_venue_matches'],
            0.5
        )
        
        self.data['team2_venue_win_rate'] = np.where(
            self.data['team2_venue_matches'] > 0,
            self.data['team2_venue_wins'] / self.data['team2_venue_matches'],
            0.5
        )
        
        print("Venue features created successfully")
        return self.data
    
    def create_toss_features(self):
        """
        Create toss-related features
        """
        print("Creating toss features...")
        
        if 'toss_winner' not in self.data.columns or 'toss_decision' not in self.data.columns:
            print("Toss columns not found, skipping toss features")
            return self.data
        
        # Toss advantage (1 if team1 wins toss, 0 if team2 wins toss, -1 if unknown)
        self.data['toss_advantage'] = np.where(
            self.data['toss_winner'] == self.data['team1'], 1,
            np.where(self.data['toss_winner'] == self.data['team2'], 0, -1)
        )
        
        # Bat first advantage (1 if toss winner chooses to bat, 0 otherwise)
        self.data['bat_first'] = (self.data['toss_decision'] == 'bat').astype(int)
        
        print("Toss features created successfully")
        return self.data
    
    def create_season_features(self):
        """
        Create season-based features if date information is available
        """
        print("Creating season features...")
        
        if 'date' in self.data.columns:
            try:
                self.data['date'] = pd.to_datetime(self.data['date'])
                self.data['year'] = self.data['date'].dt.year
                self.data['month'] = self.data['date'].dt.month
                
                # Create season feature (IPL seasons typically run from March to May)
                self.data['season'] = self.data['year']
                
                print("Season features created successfully")
            except:
                print("Could not parse date column for season features")
        else:
            print("Date column not found, skipping season features")
        
        return self.data
    
    def create_match_number_features(self):
        """
        Create match number and tournament stage features
        """
        print("Creating match number features...")
        
        if 'id' in self.data.columns:
            # Sort by match id to get chronological order
            self.data = self.data.sort_values('id')
            self.data['match_number'] = range(1, len(self.data) + 1)
            
            # Create tournament stage feature (early, middle, late stage)
            total_matches = len(self.data)
            self.data['tournament_stage'] = pd.cut(
                self.data['match_number'],
                bins=[0, total_matches//3, 2*total_matches//3, total_matches],
                labels=['early', 'middle', 'late']
            )
            
            # Encode tournament stage
            le_stage = LabelEncoder()
            self.data['tournament_stage_encoded'] = le_stage.fit_transform(self.data['tournament_stage'])
            
            print("Match number features created successfully")
        else:
            print("Match ID column not found, skipping match number features")
        
        return self.data
    
    def select_final_features(self):
        """
        Select final feature set for modeling
        """
        print("Selecting final features...")
        
        # Base features (always included)
        base_features = [
            'team1_encoded', 'team2_encoded', 'venue_encoded',
            'toss_winner_encoded', 'toss_decision_encoded'
        ]
        
        # Historical features
        historical_features = [
            'team1_win_rate', 'team2_win_rate', 'win_rate_diff',
            'head2head_team1', 'team1_matches', 'team2_matches'
        ]
        
        # Venue features
        venue_features = [
            'team1_venue_win_rate', 'team2_venue_win_rate'
        ]
        
        # Toss features
        toss_features = [
            'toss_advantage', 'bat_first'
        ]
        
        # Additional features
        additional_features = [
            'tournament_stage_encoded', 'match_number'
        ]
        
        # Combine all available features
        all_features = base_features + historical_features + venue_features + toss_features + additional_features
        available_features = [col for col in all_features if col in self.data.columns]
        
        print(f"Final feature set: {available_features}")
        return available_features
    
    def get_engineered_data(self):
        """
        Get the data with engineered features
        """
        return self.data
    
    def run_feature_engineering(self):
        """
        Run the complete feature engineering pipeline
        """
        print("="*50)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*50)
        
        self.create_historical_features()
        self.create_win_rate_features()
        self.create_venue_features()
        self.create_toss_features()
        self.create_season_features()
        self.create_match_number_features()
        
        final_features = self.select_final_features()
        
        print("="*50)
        print("FEATURE ENGINEERING COMPLETED")
        print("="*50)
        
        return self.data, final_features