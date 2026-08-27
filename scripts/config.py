# config.py
"""
Project-wide hardcoded config for use across scripts.

Latest changes: 14/08/26:
- Updated to final checkpoint/prediction dirs
"""

####################
# Data Folders
####################

DATA_DIR = '/content/drive/MyDrive/drive_diss/data'
RAW_DUMP_DIR = '/content/drive/MyDrive/drive_diss/data/raw_dumps'
GAME_DIR = '/content/drive/MyDrive/drive_diss/data/game_data'
POS_DIR = '/content/drive/MyDrive/drive_diss/data/pos_data'
IDS_DIR = '/content/drive/MyDrive/drive_diss/data/pos_data/ids'
TRAIN_DIR = '/content/drive/MyDrive/drive_diss/data/pos_data/train_data'
VAL_DIR = '/content/drive/MyDrive/drive_diss/data/pos_data/val_data'
TEST_DIR = '/content/drive/MyDrive/drive_diss/data/pos_data/test_data'
RESULTS_DIR = '/content/drive/MyDrive/drive_diss/data/results'
CHECKPOINTS_DIR = '/content/drive/MyDrive/drive_diss/checkpoints/final'
PREDICTIONS_DIR = '/content/drive/MyDrive/drive_diss/data/predictions/final'

####################
# Local Disk - Colab session 
####################

LOCAL_SPLITS_DIR = '/content/prepared_splits'

####################
# Git Dir
####################

REPO_DIR = '/content/chess-result-prediction'
REPO_SCRIPTS_DIR = '/content/chess-result-prediction/scripts'

####################
# Reproducibility
####################

SEED = 0

####################
# Stockfish
####################

STOCKFISH_PATH = '/usr/games/stockfish'