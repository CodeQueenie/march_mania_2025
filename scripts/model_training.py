"""
Model Training Module for March Madness Prediction

This module contains functions for training predictive models for the March Madness tournament.
It includes error handling and validation to ensure robust model training.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
import joblib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_feature_dataset(feature_path, validate=True):
    """
    Load a feature dataset with proper error handling.
    
    Args:
        feature_path (str): Path to the feature CSV file
        validate (bool): Whether to validate the dataset
        
    Returns:
        pd.DataFrame or None: The loaded dataset or None if error occurs
    """
    try:
        if not os.path.exists(feature_path):
            logger.error(f"Feature file not found: {feature_path}")
            return None
            
        df = pd.read_csv(feature_path)
        
        if validate:
            # Check if dataset has expected columns
            required_columns = ['Season', 'TeamID1', 'TeamID2', 'Result']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return None
                
            # Check if dataset has sufficient rows
            if len(df) < 100:
                logger.warning(f"Dataset may be too small for reliable training: {len(df)} rows")
                
        logger.info(f"Successfully loaded feature dataset with {len(df)} rows and {len(df.columns)} columns")
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
                        if col not in ['Season', 'TeamID1', 'TeamID2', 'Result', 'ID']]
        
        if not feature_cols:
            logger.error("No valid feature columns found")
            return None
            
        # Split data into features and target
        X = df[feature_cols]
        y = df['Result']
        
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
        y_train (pd.Series): Training target values
        model_type (str): Type of model to train ('random_forest', 'gradient_boosting', or 'logistic')
        cv (int): Number of cross-validation folds
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple or None: (trained_model, feature_importances, cv_results) or None if error occurs
    """
    try:
        if X_train is None or y_train is None:
            logger.error("Cannot train model with None inputs")
            return None
            
        # Create cross-validation object
        cv_folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        
        # Define the model based on type
        if model_type == 'random_forest':
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestClassifier(random_state=random_state))
            ])
            
            param_grid = {
                'model__n_estimators': [100, 200],
                'model__max_depth': [None, 10, 20],
                'model__min_samples_split': [2, 5]
            }
            
        elif model_type == 'gradient_boosting':
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', GradientBoostingClassifier(random_state=random_state))
            ])
            
            param_grid = {
                'model__n_estimators': [100, 200],
                'model__learning_rate': [0.01, 0.1],
                'model__max_depth': [3, 5]
            }
            
        elif model_type == 'logistic':
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(random_state=random_state, max_iter=1000))
            ])
            
            param_grid = {
                'model__C': [0.1, 1.0, 10.0],
                'model__penalty': ['l2'],
            }
            
        else:
            logger.error(f"Unsupported model type: {model_type}")
            return None
            
        # Train model with cross-validation
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv_folds, scoring='neg_log_loss', n_jobs=-1
        )
        
        logger.info(f"Training {model_type} model with {cv}-fold cross-validation")
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Get feature importances if available
        feature_importances = None
        if hasattr(best_model.named_steps['model'], 'feature_importances_'):
            feature_importances = pd.DataFrame({
                'feature': X_train.columns,
                'importance': best_model.named_steps['model'].feature_importances_
            }).sort_values('importance', ascending=False)
            
        logger.info(f"Model training complete. Best CV score: {-grid_search.best_score_:.4f} (log loss)")
        
        return best_model, feature_importances, grid_search.cv_results_
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        return None

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model with error handling.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target values
        
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
        log_loss_score = log_loss(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        metrics = {
            'accuracy': accuracy,
            'log_loss': log_loss_score,
            'roc_auc': roc_auc,
            'report': report
        }
        
        logger.info(f"Model evaluation complete. Accuracy: {accuracy:.4f}, Log Loss: {log_loss_score:.4f}, ROC AUC: {roc_auc:.4f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        return None

def save_model(model, model_path, metadata=None):
    """
    Save trained model to disk with error handling.
    
    Args:
        model: Trained model to save
        model_path (str): Path to save the model
        metadata (dict): Additional metadata to save with the model
        
    Returns:
        bool: True if save successful, False otherwise
    """
    try:
        if model is None:
            logger.error("Cannot save None model")
            return False
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model with metadata
        save_dict = {
            'model': model,
            'metadata': metadata if metadata is not None else {}
        }
        joblib.dump(save_dict, model_path)
        
        logger.info(f"Model successfully saved to {model_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        return False

def load_model(model_path):
    """
    Load a trained model from disk with error handling.
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        tuple or None: (model, metadata) or None if error occurs
    """
    try:
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None
            
        # Load model and metadata
        save_dict = joblib.load(model_path)
        model = save_dict['model']
        metadata = save_dict.get('metadata', {})
        
        logger.info(f"Model successfully loaded from {model_path}")
        return model, metadata
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def generate_tournament_predictions(model, features_df, output_path=None, proba_threshold=0.5):
    """
    Generate predictions for tournament matchups.
    
    Args:
        model: Trained model
        features_df (pd.DataFrame): Features for tournament matchups
        output_path (str): Path to save predictions (optional)
        proba_threshold (float): Probability threshold for binary predictions
        
    Returns:
        pd.DataFrame or None: DataFrame with predictions or None if error occurs
    """
    try:
        if model is None or features_df is None or features_df.empty:
            logger.error("Cannot generate predictions with None or empty inputs")
            return None
            
        # Identify feature columns
        id_columns = ['Season', 'TeamID1', 'TeamID2']
        feature_cols = [col for col in features_df.columns if col not in id_columns and col != 'Result']
        
        if not all(col in features_df.columns for col in id_columns):
            logger.error(f"Missing required ID columns in features dataframe")
            return None
            
        if not feature_cols:
            logger.error("No valid feature columns found")
            return None
            
        # Generate predictions
        X_pred = features_df[feature_cols]
        predictions_proba = model.predict_proba(X_pred)[:, 1]
        predictions_binary = (predictions_proba > proba_threshold).astype(int)
        
        # Create prediction dataframe
        results_df = features_df[id_columns].copy()
        results_df['Pred'] = predictions_proba
        results_df['PredBinary'] = predictions_binary
        
        # Create submission format ID
        results_df['ID'] = results_df['Season'].astype(str) + '_' + \
                          results_df['TeamID1'].astype(str) + '_' + \
                          results_df['TeamID2'].astype(str)
                          
        # Keep only required columns for submission
        submission_df = results_df[['ID', 'Pred']].copy()
        
        # Save to file if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            submission_df.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to {output_path}")
        
        logger.info(f"Generated predictions for {len(features_df)} matchups")
        return submission_df
        
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        return None
