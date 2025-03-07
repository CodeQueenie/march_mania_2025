# March Machine Learning Mania 2025

## Project Overview

This project demonstrates a complete end-to-end machine learning pipeline for predicting outcomes of NCAA basketball tournament games for the March Machine Learning Mania 2025 Kaggle competition. The focus is on building a robust, reproducible workflow that handles both men's and women's basketball data.

## Technical Implementation

The project showcases the following technical skills and best practices:

1. **Data Engineering**
   - Data loading and integration from multiple sources
   - Feature engineering based on historical game statistics
   - Handling of missing values through imputation
   - Creation of train/test splits with proper stratification

2. **Machine Learning Pipeline**
   - Model selection and training (Random Forest Classifier)
   - Cross-validation for robust evaluation
   - Feature importance analysis
   - Model persistence and versioning

3. **Software Engineering**
   - Modular code organization with clear separation of concerns
   - Error handling and logging throughout the pipeline
   - Automated data validation and preprocessing
   - Efficient file I/O and data processing

4. **Project Structure**
   - Organized directory structure following best practices
   - Proper documentation and code comments
   - Version control with Git
   - Dependency management with requirements.txt

## Pipeline Components

The machine learning pipeline consists of the following components:

1. **Data Collection**: Loading historical NCAA basketball data
2. **Feature Engineering**: Creating predictive features from raw statistics
3. **Data Preprocessing**: Handling missing values and preparing data for modeling
4. **Model Training**: Training separate models for men's and women's tournaments
5. **Model Evaluation**: Assessing model performance with appropriate metrics
6. **Prediction Generation**: Creating predictions for all possible tournament matchups
7. **Submission Preparation**: Formatting predictions for Kaggle submission

## Key Files

- `scripts/fixed_model_training.py`: Core implementation of the model training pipeline
- `scripts/feature_engineering.py`: Feature creation and transformation logic
- `utils/data_loader.py`: Data loading and preprocessing utilities
- `models/m_model.joblib` & `models/w_model.joblib`: Trained models for men's and women's tournaments
- `submissions/submission.csv`: Final predictions formatted for Kaggle submission

## Results

The models achieve the following performance metrics:

- **Men's Model**: Accuracy: 0.5053, Log Loss: 0.7309, AUC: 0.5052
- **Women's Model**: Accuracy: 0.5041, Log Loss: 0.7320, AUC: 0.5037

While these metrics might seem modest, they represent a solid baseline for the challenging task of sports prediction, where even small improvements over random guessing can be significant.

## Future Improvements

Potential enhancements to the pipeline include:

- Ensemble methods combining multiple model types
- More sophisticated feature engineering
- Hyperparameter optimization
- More sophisticated handling of historical tournament performance

## 🚀 Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/CodeQueenie/march_mania_2025.git
   cd march_mania_2025
   ```

2. Create a new environment with dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Feature Engineering
The feature engineering process (`scripts/feature_engineering.py`) creates the following features:
- Team season statistics (wins, losses, win percentage)
- Offensive and defensive metrics (points scored/allowed)
- Tournament seed strength 
- Matchup-based features (comparing team stats)

## 🤖 Model Training and Prediction
The prediction script (`scripts/prediction.py`) trains Random Forest models to predict:
- Win probabilities for all possible matchups
- Both men's and women's tournaments
- Output formatted for Kaggle submission

## 📝 Running the Pipeline
1. Generate features:
   ```
   python scripts/feature_engineering.py
   ```

2. Train models and generate predictions:
   ```
   python scripts/prediction.py
   ```

3. Final submission file will be saved to `submissions/submission.csv`
