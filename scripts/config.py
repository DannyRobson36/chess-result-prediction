"""
config.py
Project-wide config: fixed structural constants plus environment-specific
values loaded from .env.

Latest changes: 17/09/26:
- Switched environment-specific values to .env
"""

import os

from dotenv import load_dotenv

load_dotenv()

####################
# Environment (.env)
####################

PROJECT_NAME = os.environ['PROJECT_NAME']
GITHUB_USER = os.environ['GITHUB_USER']
REPO_NAME = os.environ['REPO_NAME']
STOCKFISH_PATH = os.environ['STOCKFISH_PATH']
OVERLEAF_PROJECT_ID = os.environ['OVERLEAF_PROJECT_ID']

####################
# Drive Root
####################

DRIVE_ROOT = '/content/drive/MyDrive'
PROJECT_ROOT = f'{DRIVE_ROOT}/{PROJECT_NAME}'

####################
# Data Folders
####################

DATA_DIR = f'{PROJECT_ROOT}/data'
RAW_DUMP_DIR = f'{DATA_DIR}/raw_dumps'
GAME_DIR = f'{DATA_DIR}/game_data'
POS_DIR = f'{DATA_DIR}/pos_data'
IDS_DIR = f'{POS_DIR}/ids'
TRAIN_DIR = f'{POS_DIR}/train_data'
VAL_DIR = f'{POS_DIR}/val_data'
TEST_DIR = f'{POS_DIR}/test_data'
RESULTS_DIR = f'{DATA_DIR}/results'
CHECKPOINTS_DIR = f'{PROJECT_ROOT}/checkpoints/final'
PREDICTIONS_DIR = f'{DATA_DIR}/predictions/final'

####################
# Local Disk
####################

LOCAL_SPLITS_DIR = '/content/prepared_splits'

####################
# Git Dir
####################

REPO_DIR = f'/content/{REPO_NAME}'
REPO_SCRIPTS_DIR = f'{REPO_DIR}/scripts'

####################
# Overleaf
####################

OVERLEAF_DIR = f'{PROJECT_ROOT}/overleaf_diss'
FIGURES_EDA = f'{OVERLEAF_DIR}/figures/eda'
FIGURES_TRAINING = f'{OVERLEAF_DIR}/figures/training'
FIGURES_RESULTS = f'{OVERLEAF_DIR}/figures/results'
TABLES_EDA = f'{OVERLEAF_DIR}/tables/eda'
TABLES_TRAINING = f'{OVERLEAF_DIR}/tables/training'
TABLES_RESULTS = f'{OVERLEAF_DIR}/tables/results'

####################
# Reproducibility
####################

SEED = 0