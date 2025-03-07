# Cleanup script for March Mania 2025 project

# Remove redundant notebooks
Remove-Item -Path "notebooks/Untitled.ipynb" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "notebooks/eda_V*.ipynb" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "notebooks/error_log.txt" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "notebooks/fixed_notebook_section.py" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "notebooks/notebook_helper.py" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "notebooks/notebook_section.py" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "notebooks/pipeline_demo.ipynb" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "notebooks/portfolio_enhancements.md" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "notebooks/simple_model_training.ipynb" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "notebooks/simple_model_training.py" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "notebooks/test_fixed_model.py" -Force -ErrorAction SilentlyContinue

# Remove redundant models
Remove-Item -Path "models/men_imputer.joblib" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "models/men_scaler.joblib" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "models/women_imputer.joblib" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "models/women_scaler.joblib" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "models/men_model.joblib" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "models/women_model.joblib" -Force -ErrorAction SilentlyContinue

# Remove redundant submissions
Remove-Item -Path "submissions/men_predictions.csv" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "submissions/women_predictions.csv" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "submissions/submission.csv" -Force -ErrorAction SilentlyContinue

# Rename combined_predictions.csv to submission.csv for Kaggle
Copy-Item -Path "submissions/combined_predictions.csv" -Destination "submissions/submission.csv" -Force
Write-Host "Project cleaned up successfully!"
