import pandas as pd
import numpy as np
import os
import pickle
import sys
import joblib

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from utils.logger import app_logger as logger
from utils.data_loader import DataLoader

# Initialize data loader with correct paths
data_dir = os.path.join(project_root, 'data')
cache_dir = os.path.join(project_root, 'cache')
data_loader = DataLoader(data_dir=data_dir, cache_dir=cache_dir)

def train_model(df_features, target_column=None, gender='M'):
    """
    Train a model using the generated features.
    
    Args:
        df_features (pd.DataFrame): DataFrame with features
        target_column (str): Column name for the target (None for submission predictions)
        gender (str): 'M' for men's data, 'W' for women's data
        
    Returns:
        tuple: (model, scaler, imputer) - trained model and preprocessing objects
    """
    logger.info(f"Training model for {gender} tournament")
    
    # If target column is provided, use it for training
    if target_column is not None and target_column in df_features.columns:
        # Features and target
        X = df_features.drop(['ID', target_column, 'Season', 'Team1ID', 'Team2ID'], axis=1)
        y = df_features[target_column]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        logger.info(f"Model training complete - Training score: {train_score:.4f}, Testing score: {test_score:.4f}")
    else:
        # For submission, train on all available data
        # For demonstration, we'll create a simple model
        
        # Features (exclude non-feature columns)
        columns_to_drop = ['ID', 'Season', 'Team1ID', 'Team2ID']
        feature_columns = [col for col in df_features.columns if col not in columns_to_drop]
        X = df_features[feature_columns]
        
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Train a simple model (in reality, you'd want to train on historical outcomes)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Since we don't have labels for submission, we'll just fit on synthetic data for demo
        # In reality, you would use historical tournament results to train your model
        y_synthetic = np.random.randint(0, 2, size=X.shape[0])
        model.fit(X, y_synthetic)
        
        logger.info(f"Model created for {gender} submission predictions")
    
    return model, scaler, imputer

def generate_predictions(df_features, model, scaler, imputer):
    """
    Generate predictions for the given features.
    
    Args:
        df_features (pd.DataFrame): DataFrame with features
        model: Trained model
        scaler: Fitted scaler
        imputer: Fitted imputer
        
    Returns:
        pd.DataFrame: DataFrame with ID and prediction probability
    """
    # Extract IDs
    ids = df_features['ID'].values
    
    # Prepare features
    columns_to_drop = ['ID', 'Season', 'Team1ID', 'Team2ID']
    feature_columns = [col for col in df_features.columns if col not in columns_to_drop]
    X = df_features[feature_columns]
    
    # Handle missing values
    X = imputer.transform(X)
    
    # Scale features
    X = scaler.transform(X)
    
    # Generate predictions
    probs = model.predict_proba(X)[:, 1]  # Probability of class 1 (lower ID team wins)
    
    # Create submission DataFrame
    df_submission = pd.DataFrame({
        'ID': ids,
        'Pred': probs
    })
    
    return df_submission

def main():
    """Generate predictions for both men's and women's tournaments and create submission file"""
    # Directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    feature_dir = os.path.join(project_dir, 'data', 'features')
    model_dir = os.path.join(project_dir, 'models')
    output_dir = os.path.join(project_dir, 'submissions')
    
    # Create directories if they don't exist
    for directory in [feature_dir, model_dir, output_dir]:
        os.makedirs(directory, exist_ok=True)
    
    submissions = []
    
    # Process men's tournament
    men_features_path = os.path.join(feature_dir, 'men_features.csv')
    if os.path.exists(men_features_path):
        logger.info("Processing men's tournament predictions")
        df_men_features = pd.read_csv(men_features_path)
        
        # Train model
        men_model, men_scaler, men_imputer = train_model(df_men_features, gender='M')
        
        # Save model and preprocessing objects
        joblib.dump(men_model, os.path.join(model_dir, 'men_model.joblib'))
        joblib.dump(men_scaler, os.path.join(model_dir, 'men_scaler.joblib'))
        joblib.dump(men_imputer, os.path.join(model_dir, 'men_imputer.joblib'))
        
        # Generate predictions
        df_men_submission = generate_predictions(df_men_features, men_model, men_scaler, men_imputer)
        submissions.append(df_men_submission)
    else:
        logger.warning(f"Men's features file not found at {men_features_path}")
    
    # Process women's tournament
    women_features_path = os.path.join(feature_dir, 'women_features.csv')
    if os.path.exists(women_features_path):
        logger.info("Processing women's tournament predictions")
        df_women_features = pd.read_csv(women_features_path)
        
        # Train model
        women_model, women_scaler, women_imputer = train_model(df_women_features, gender='W')
        
        # Save model and preprocessing objects
        joblib.dump(women_model, os.path.join(model_dir, 'women_model.joblib'))
        joblib.dump(women_scaler, os.path.join(model_dir, 'women_scaler.joblib'))
        joblib.dump(women_imputer, os.path.join(model_dir, 'women_imputer.joblib'))
        
        # Generate predictions
        df_women_submission = generate_predictions(df_women_features, women_model, women_scaler, women_imputer)
        submissions.append(df_women_submission)
    else:
        logger.warning(f"Women's features file not found at {women_features_path}")
    
    # Combine submissions
    if submissions:
        df_submission = pd.concat(submissions)
        
        # Save submission file
        submission_path = os.path.join(output_dir, 'submission.csv')
        df_submission.to_csv(submission_path, index=False)
        logger.info(f"Submission file saved to {submission_path}")
    else:
        logger.error("No features found for prediction")

if __name__ == "__main__":
    main()
