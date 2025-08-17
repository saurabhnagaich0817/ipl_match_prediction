import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class IPLModelTrainer:
    def __init__(self):
        """
        Initialize the model trainer
        """
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_data(self, X, y, scale_features=True):
        """
        Prepare data for modeling
        """
        print("Preparing data for modeling...")
        
        self.feature_names = X.columns.tolist()
        
        if scale_features:
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            print("Features scaled using StandardScaler")
            return X_scaled, y
        
        return X, y
    
    def initialize_models(self):
        """
        Initialize different models to try
        """
        self.models = {
            'Logistic_Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            ),
            'Random_Forest': RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                class_weight='balanced'
            ),
            'Gradient_Boosting': GradientBoostingClassifier(
                random_state=42,
                n_estimators=100
            )
        }
        
        print(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
        return self.models
    
    def evaluate_models_cv(self, X, y, cv_folds=5):
        """
        Evaluate all models using cross-validation
        """
        print("="*50)
        print("MODEL EVALUATION WITH CROSS-VALIDATION")
        print("="*50)
        
        if not self.models:
            self.initialize_models()
        
        results = {}
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            cv_precision = cross_val_score(model, X, y, cv=cv, scoring='precision')
            cv_recall = cross_val_score(model, X, y, cv=cv, scoring='recall')
            cv_f1 = cross_val_score(model, X, y, cv=cv, scoring='f1')
            cv_roc_auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            
            results[name] = {
                'accuracy': cv_scores,
                'precision': cv_precision,
                'recall': cv_recall,
                'f1': cv_f1,
                'roc_auc': cv_roc_auc
            }
            
            print(f"Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"Precision: {cv_precision.mean():.4f} (+/- {cv_precision.std() * 2:.4f})")
            print(f"Recall: {cv_recall.mean():.4f} (+/- {cv_recall.std() * 2:.4f})")
            print(f"F1-Score: {cv_f1.mean():.4f} (+/- {cv_f1.std() * 2:.4f})")
            print(f"ROC-AUC: {cv_roc_auc.mean():.4f} (+/- {cv_roc_auc.std() * 2:.4f})")
        
        return results
    
    def hyperparameter_tuning(self, X, y, model_name='Logistic_Regression'):
        """
        Perform hyperparameter tuning for the specified model
        """
        print(f"\nPerforming hyperparameter tuning for {model_name}...")
        
        param_grids = {
            'Logistic_Regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'Random_Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Gradient_Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
        
        if model_name not in param_grids:
            print(f"No parameter grid defined for {model_name}")
            return None
        
        model = self.models[model_name]
        param_grid = param_grids[model_name]
        
        # Reduce parameter grid for faster execution
        if model_name == 'Random_Forest':
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [None, 10],
                'min_samples_split': [2, 5]
            }
        elif model_name == 'Gradient_Boosting':
            param_grid = {
                'n_estimators': [50, 100],
                'learning_rate': [0.1, 0.2],
                'max_depth': [3, 5]
            }
        
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='accuracy', n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_best_model(self, X, y, model_name='Logistic_Regression'):
        """
        Train the best model after hyperparameter tuning
        """
        print(f"\nTraining best {model_name} model...")
        
        # Perform hyperparameter tuning
        best_model = self.hyperparameter_tuning(X, y, model_name)
        
        if best_model is None:
            print("Using default model parameters")
            best_model = self.models[model_name]
        
        # Train the model on full dataset
        best_model.fit(X, y)
        
        self.best_model = best_model
        self.best_model_name = model_name
        
        print(f"Best {model_name} model trained successfully")
        return best_model
    
    def evaluate_final_model(self, X_train, y_train, X_test, y_test):
        """
        Evaluate the final model on test set
        """
        if self.best_model is None:
            print("No trained model available. Please train a model first.")
            return None
        
        print("="*50)
        print("FINAL MODEL EVALUATION")
        print("="*50)
        
        # Predictions
        y_train_pred = self.best_model.predict(X_train)
        y_test_pred = self.best_model.predict(X_test)
        
        y_train_proba = self.best_model.predict_proba(X_train)[:, 1]
        y_test_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # Training performance
        print("\nTRAINING SET PERFORMANCE:")
        print(classification_report(y_train, y_train_pred))
        print(f"ROC-AUC Score: {roc_auc_score(y_train, y_train_proba):.4f}")
        
        # Test performance
        print("\nTEST SET PERFORMANCE:")
        print(classification_report(y_test, y_test_pred))
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_test_proba):.4f}")
        
        # Confusion Matrix
        self.plot_confusion_matrix(y_test, y_test_pred)
        
        # Feature importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            self.plot_feature_importance()
        elif hasattr(self.best_model, 'coef_'):
            self.plot_feature_coefficients()
        
        return {
            'train_score': roc_auc_score(y_train, y_train_proba),
            'test_score': roc_auc_score(y_test, y_test_proba),
            'predictions': y_test_pred,
            'probabilities': y_test_proba
        }
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plot confusion matrix
        """
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Team2 Wins', 'Team1 Wins'],
                    yticklabels=['Team2 Wins', 'Team1 Wins'])
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self):
        """
        Plot feature importance for tree-based models
        """
        if self.feature_names is None:
            print("Feature names not available")
            return
        
        importances = self.best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title(f"Feature Importance - {self.best_model_name}")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), 
                   [self.feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_coefficients(self):
        """
        Plot feature coefficients for linear models
        """
        if self.feature_names is None:
            print("Feature names not available")
            return
        
        coefficients = self.best_model.coef_[0]
        indices = np.argsort(np.abs(coefficients))[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title(f"Feature Coefficients - {self.best_model_name}")
        colors = ['red' if coef < 0 else 'blue' for coef in coefficients[indices]]
        plt.bar(range(len(coefficients)), coefficients[indices], color=colors)
        plt.xticks(range(len(coefficients)), 
                   [self.feature_names[i] for i in indices], rotation=45)
        plt.ylabel('Coefficient Value')
        plt.tight_layout()
        plt.savefig('feature_coefficients.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath='ipl_prediction_model.pkl'):
        """
        Save the trained model
        """
        if self.best_model is None:
            print("No trained model to save")
            return
        
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_name': self.best_model_name
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='ipl_prediction_model.pkl'):
        """
        Load a trained model
        """
        try:
            model_data = joblib.load(filepath)
            self.best_model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.best_model_name = model_data['model_name']
            print(f"Model loaded from {filepath}")
            return True
        except FileNotFoundError:
            print(f"Model file {filepath} not found")
            return False
    
    def predict_match(self, team1, team2, venue, toss_winner, toss_decision, 
                     team1_features=None, team2_features=None):
        """
        Predict the outcome of a single match
        """
        if self.best_model is None:
            print("No trained model available")
            return None
        
        print(f"\nPredicting match: {team1} vs {team2}")
        print(f"Venue: {venue}")
        print(f"Toss: {toss_winner} chose to {toss_decision}")
        
        # This is a simplified prediction function
        # In a real scenario, you would need to encode the inputs properly
        # and include all the engineered features
        
        # Create a basic feature vector (this would need to be expanded)
        features = np.array([[0, 1, 0, 1, 1]])  # Placeholder values
        
        prediction = self.best_model.predict(features)[0]
        probability = self.best_model.predict_proba(features)[0]
        
        winner = team1 if prediction == 1 else team2
        confidence = max(probability)
        
        print(f"Predicted Winner: {winner}")
        print(f"Confidence: {confidence:.2%}")
        
        return {
            'predicted_winner': winner,
            'confidence': confidence,
            'team1_win_prob': probability[1],
            'team2_win_prob': probability[0]
        }
    
    def model_summary(self):
        """
        Print model summary
        """
        if self.best_model is None:
            print("No trained model available")
            return
        
        print("="*50)
        print("MODEL SUMMARY")
        print("="*50)
        print(f"Best Model: {self.best_model_name}")
        print(f"Features Used: {len(self.feature_names)}")
        print(f"Feature Names: {self.feature_names}")
        print(f"Model Parameters: {self.best_model.get_params()}")
        print("="*50)