# Titanic: Machine Learning from Disaster

This directory contains code and resources for the Kaggle Titanic competition.

## Competition Overview

The Titanic competition is one of the most popular introductory machine learning competitions on Kaggle. The goal is to predict which passengers survived the Titanic shipwreck based on features like age, sex, passenger class, and more.

**Competition Link**: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

## Directory Structure

```
competitions/titanic/
├── README.md               # This file
├── data/                   # Data directory
│   ├── train.csv           # Training data
│   ├── test.csv            # Test data
│   └── gender_submission.csv  # Sample submission file
├── submissions/            # Submission files
└── notebooks/              # Jupyter notebooks
```

## Getting Started

1. Ensure you have set up your Kaggle API credentials (see the main project README)
2. Download the competition data if you haven't already:
   ```
   python utils/kaggle_utils.py download titanic
   ```
3. Run the model training script:
   ```
   python titanic_model.py
   ```
4. Check the `submissions` directory for the generated submission file

## Model Approach

Our approach includes:

1. **Data preprocessing**:
   - Missing value imputation (Age, Fare, Embarked)
   - Feature engineering (Title extraction, family size, etc.)
   - Categorical encoding

2. **Model Selection**:
   - Logistic Regression (baseline)
   - Random Forest (our primary model)

3. **Evaluation**:
   - Cross-validation
   - Performance metrics (accuracy, precision, recall, F1)

## Files

- `titanic_model.py`: Main Python script for training the model and generating a submission
- Configuration settings are in `utils/config.py`

## Results

Our Random Forest model achieves approximately 78-80% accuracy on cross-validation.

## Future Improvements

- Hyperparameter tuning
- Additional feature engineering
- Model ensembling
- More advanced models (XGBoost, LightGBM, Neural Networks)
