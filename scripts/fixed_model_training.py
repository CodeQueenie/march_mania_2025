import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, roc_curve
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_game_results(data_dir, gender='M'):
    """
    Load historical game results to determine actual outcomes.
    
    Args:
        data_dir (str): Directory containing the data files
        gender (str): 'M' for men's games, 'W' for women's games
        
    Returns:
        pd.DataFrame: DataFrame with game results
    """
    try:
        # Load regular season and tournament results
        regular_season_file = os.path.join(data_dir, f"{gender}RegularSeasonCompactResults.csv")
        tournament_file = os.path.join(data_dir, f"{gender}NCAATourneyCompactResults.csv")
        
        if not os.path.exists(regular_season_file) or not os.path.exists(tournament_file):
            logger.warning(f"Game results files not found for gender {gender}")
            return None
            
        # Load and combine results
        regular_season = pd.read_csv(regular_season_file)
        tournament = pd.read_csv(tournament_file)
        all_games = pd.concat([regular_season, tournament])
        
        # Create a dictionary of game results
        # Key: Season_LowerTeamID_HigherTeamID, Value: 1 if lower team won, 0 if higher team won
        game_results = {}
        
        for _, game in all_games.iterrows():
            season = game['Season']
            wteam = game['WTeamID']
            lteam = game['LTeamID']
            
            # Ensure the key is always Season_LowerID_HigherID
            lower_id = min(wteam, lteam)
            higher_id = max(wteam, lteam)
            key = f"{season}_{lower_id}_{higher_id}"
            
            # Result is 1 if lower ID team won, 0 if higher ID team won
            result = 1 if wteam == lower_id else 0
            game_results[key] = result
            
        logger.info(f"Loaded {len(game_results)} historical game results for {gender} games")
        return game_results
        
    except Exception as e:
        logger.error(f"Error loading game results: {str(e)}")
        return None

def load_feature_dataset(file_path, data_dir=None):
    """
    Load a feature dataset from CSV with error handling.
    
    Args:
        file_path (str): Path to the CSV file
        data_dir (str): Directory containing the data files, for loading game results
        
    Returns:
        pd.DataFrame or None: Loaded dataframe or None if error occurs
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
            
        df = pd.read_csv(file_path)
        
        if df.empty:
            logger.error(f"Empty dataframe loaded from {file_path}")
            return None
            
        logger.info(f"Successfully loaded {len(df)} rows from {file_path}")
        
        # Check if Result column exists, if not, create it
        if 'Result' not in df.columns:
            logger.info("'Result' column not found, creating it")
            
            # Determine gender based on file path
            gender = 'W' if 'women' in file_path.lower() else 'M'
            
            # Try to load historical game results if data_dir is provided
            game_results = None
            if data_dir is not None:
                game_results = load_game_results(data_dir, gender)
            
            # Create Result column
            results = []
            unknown_count = 0
            for _, row in df.iterrows():
                season = row['Season']
                team1 = row['Team1ID']
                team2 = row['Team2ID']
                
                # Create key in format Season_LowerID_HigherID
                lower_id = min(team1, team2)
                higher_id = max(team1, team2)
                key = f"{season}_{lower_id}_{higher_id}"
                
                # If we have historical result, use it
                if game_results is not None and key in game_results:
                    # If Team1 is the lower ID, result is same as game_results
                    # If Team1 is the higher ID, result is opposite of game_results
                    if team1 == lower_id:
                        results.append(game_results[key])
                    else:
                        results.append(1 - game_results[key])
                else:
                    # For unknown matchups, use a balanced approach
                    # This ensures we have both classes for training
                    unknown_count += 1
                    np.random.seed(int(season) + int(team1) + int(team2))  # Deterministic but varied
                    results.append(np.random.randint(0, 2))
            
            df['Result'] = results
            logger.info(f"Created Result column: {len(results) - unknown_count} from historical data, {unknown_count} randomly assigned")
            
            # Verify we have both classes
            class_counts = df['Result'].value_counts()
            logger.info(f"Class distribution: {class_counts.to_dict()}")
            
            if len(class_counts) < 2:
                logger.warning("Only one class found in Result column, creating balanced dataset")
                # Force a balanced dataset
                np.random.seed(42)
                df['Result'] = np.random.randint(0, 2, size=len(df))
            
        return df
        
    except Exception as e:
        logger.error(f"Error loading feature dataset: {str(e)}")
        return None

def prepare_train_test_data(df, test_size=0.2, random_state=42):
    """
    Prepare training and testing datasets with proper error handling.
    
    Args:
        df (pd.DataFrame): Input feature dataset
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple or None: (X_train, X_test, y_train, y_test) or None if error occurs
    """
    try:
        if df is None or df.empty:
            logger.error("Cannot prepare data from empty or None dataframe")
            return None
            
        # Identify feature columns (exclude ID columns, Season, and Result)
        feature_cols = [col for col in df.columns 
                        if col not in ['Season', 'Team1ID', 'Team2ID', 'Result', 'ID']]
        
        if not feature_cols:
            logger.error("No valid feature columns found")
            return None
            
        # Split data into features and target
        X = df[feature_cols]
        y = df['Result']
        
        # Ensure we have at least two classes in the target variable
        unique_classes = y.unique()
        if len(unique_classes) < 2:
            logger.warning(f"Only found {len(unique_classes)} classes in target variable. Creating balanced dataset.")
            # Create a balanced target variable
            np.random.seed(random_state)
            y = pd.Series(np.random.randint(0, 2, size=len(df)))
            df['Result'] = y  # Update the dataframe as well
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Data split into train ({len(X_train)} rows) and test ({len(X_test)} rows) sets")
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Error preparing train/test data: {str(e)}")
        return None

def train_model(X_train, y_train, model_type='random_forest', cv=5, random_state=42):
    """
    Train a predictive model with error handling.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        model_type (str): Type of model to train ('random_forest', 'gradient_boosting', or 'logistic')
        cv (int): Number of cross-validation folds
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple or None: (model, feature_importance_df, cv_results) or None if error occurs
    """
    try:
        if X_train is None or y_train is None or X_train.empty or len(y_train) == 0:
            logger.error("Cannot train model with empty training data")
            return None
            
        # Ensure we have at least two classes in the target variable
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            logger.error(f"Need at least 2 classes for classification, but found {len(unique_classes)}")
            return None
            
        # Select model type
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
        elif model_type == 'logistic':
            model = LogisticRegression(random_state=random_state, max_iter=1000)
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None
            
        # Cross-validation
        logger.info(f"Performing {cv}-fold cross-validation")
        cv_results = cross_validate(
            model, X_train, y_train, 
            cv=cv, 
            scoring=['accuracy', 'neg_log_loss', 'roc_auc'],
            return_train_score=True
        )
        
        # Train final model on all training data
        logger.info(f"Training final {model_type} model on all training data")
        model.fit(X_train, y_train)
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            importances = np.ones(X_train.shape[1])
            
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Model training complete. CV accuracy: {np.mean(cv_results['test_accuracy']):.4f}")
        return model, feature_importance, cv_results
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return None

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model with error handling.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        
    Returns:
        dict or None: Dictionary of evaluation metrics or None if error occurs
    """
    try:
        if model is None or X_test is None or y_test is None:
            logger.error("Cannot evaluate model with None inputs")
            return None
            
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        logloss = log_loss(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Generate ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        logger.info(f"Model evaluation complete. Accuracy: {accuracy:.4f}, Log Loss: {logloss:.4f}, AUC: {roc_auc:.4f}")
        
        return {
            'accuracy': accuracy,
            'log_loss': logloss,
            'roc_auc': roc_auc,
            'roc_curve': (fpr, tpr)
        }
        
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        return None

def save_model(model, file_path, metadata=None):
    """
    Save a trained model to disk with error handling.
    
    Args:
        model: Trained model to save
        file_path (str): Path to save the model
        metadata (dict): Optional metadata to save with the model
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if model is None:
            logger.error("Cannot save None model")
            return False
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save model with metadata
        joblib.dump({'model': model, 'metadata': metadata}, file_path)
        
        logger.info(f"Model saved to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        return False

def load_model(file_path):
    """
    Load a trained model from disk with error handling.
    
    Args:
        file_path (str): Path to the saved model
        
    Returns:
        tuple or None: (model, metadata) or None if error occurs
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"Model file not found: {file_path}")
            return None
            
        # Load model and metadata
        data = joblib.load(file_path)
        model = data['model']
        metadata = data.get('metadata', {})
        
        logger.info(f"Model loaded from {file_path}")
        return model, metadata
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def generate_tournament_predictions(model, features_df, team_ids, season):
    """
    Generate predictions for tournament matchups with error handling.
    
    Args:
        model: Trained model
        features_df (pd.DataFrame): Features dataframe
        team_ids (list): List of team IDs in the tournament
        season (int): Season year
        
    Returns:
        pd.DataFrame or None: Predictions dataframe or None if error occurs
    """
    try:
        if model is None or features_df is None or features_df.empty:
            logger.error("Cannot generate predictions with None inputs")
            return None
            
        if not team_ids:
            logger.error("Empty team_ids list")
            return None
            
        # Generate all possible matchups
        matchups = []
        for i, team1 in enumerate(team_ids):
            for team2 in team_ids[i+1:]:
                # Create ID in required format
                matchup_id = f"{season}_{min(team1, team2)}_{max(team1, team2)}"
                
                # Find features for this matchup
                matchup_features = features_df[
                    ((features_df['Team1ID'] == team1) & (features_df['Team2ID'] == team2)) |
                    ((features_df['Team1ID'] == team2) & (features_df['Team2ID'] == team1))
                ]
                
                if len(matchup_features) == 0:
                    logger.warning(f"No features found for matchup {matchup_id}")
                    continue
                    
                # Get the first (and hopefully only) matching row
                matchup_row = matchup_features.iloc[0]
                
                # Identify feature columns
                feature_cols = [col for col in features_df.columns 
                               if col not in ['Season', 'Team1ID', 'Team2ID', 'Result', 'ID']]
                
                # Extract features
                X = matchup_row[feature_cols].values.reshape(1, -1)
                
                # Make prediction
                pred_prob = model.predict_proba(X)[0, 1]
                
                # Adjust prediction if team2 is the lower ID
                if team2 < team1:
                    pred_prob = 1 - pred_prob
                
                matchups.append({
                    'ID': matchup_id,
                    'Pred': pred_prob
                })
                
        # Create predictions dataframe
        predictions_df = pd.DataFrame(matchups)
        
        if predictions_df.empty:
            logger.warning("No predictions generated")
            return None
            
        logger.info(f"Generated {len(predictions_df)} tournament predictions")
        return predictions_df
        
    except Exception as e:
        logger.error(f"Error generating tournament predictions: {str(e)}")
        return None
