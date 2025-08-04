"""
Configuration settings for Kaggle projects.
Modify these settings as needed for each competition.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
COMPETITION_DIR = os.path.join(BASE_DIR, 'competitions')
DATASET_DIR = os.path.join(BASE_DIR, 'datasets')
NOTEBOOK_DIR = os.path.join(BASE_DIR, 'notebooks')
SUBMISSION_DIR = os.path.join(BASE_DIR, 'submissions')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Current competition (change this for each competition)
CURRENT_COMPETITION = "titanic"

# Paths for the current competition
COMPETITION_DATA_DIR = os.path.join(DATASET_DIR, CURRENT_COMPETITION)
COMPETITION_MODEL_DIR = os.path.join(MODEL_DIR, CURRENT_COMPETITION)
COMPETITION_NOTEBOOK_DIR = os.path.join(NOTEBOOK_DIR, CURRENT_COMPETITION)
COMPETITION_SUBMISSION_DIR = os.path.join(SUBMISSION_DIR, CURRENT_COMPETITION)

# Create directories if they don't exist
for directory in [COMPETITION_DATA_DIR, COMPETITION_MODEL_DIR, 
                  COMPETITION_NOTEBOOK_DIR, COMPETITION_SUBMISSION_DIR]:
    os.makedirs(directory, exist_ok=True)

# File paths (modify as needed for each competition)
TRAIN_DATA_PATH = os.path.join(COMPETITION_DATA_DIR, 'train.csv')
TEST_DATA_PATH = os.path.join(COMPETITION_DATA_DIR, 'test.csv')
SAMPLE_SUBMISSION_PATH = os.path.join(COMPETITION_DATA_DIR, 'gender_submission.csv')

# Model settings
RANDOM_STATE = 42
VALIDATION_SIZE = 0.2
N_FOLDS = 5

# Feature engineering settings - Titanic specific
CATEGORICAL_COLS = ['Sex', 'Embarked', 'Pclass']  
NUMERICAL_COLS = ['Age', 'SibSp', 'Parch', 'Fare']  
TARGET_COL = 'Survived'  
ID_COL = 'PassengerId'  

# Hyperparameter optimization
N_TRIALS = 50  # Reduced for faster execution
N_JOBS = -1  # Use all available cores

# Logging settings
LOG_LEVEL = 'INFO'
EXPERIMENT_TRACKING = False  # Set to True to enable experiment tracking

# Additional Titanic-specific settings
FEATURE_ENGINEERING = {
    'age_imputation': 'median',  # Strategy for imputing missing age values
    'create_family_size': True,  # Create a family size feature (SibSp + Parch + 1)
    'extract_title': True,       # Extract title from Name (Mr, Mrs, Miss, etc.)
    'bin_age': True,             # Create age groups
    'fare_imputation': 'median', # Strategy for imputing missing fare values
    'embarked_imputation': 'mode' # Strategy for imputing missing embarked values
}

# Kaggle API settings (set these as environment variables in production)
# KAGGLE_USERNAME = os.environ.get('KAGGLE_USERNAME')
# KAGGLE_KEY = os.environ.get('KAGGLE_KEY') 