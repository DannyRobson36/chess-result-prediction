# config.py
"""
Project-wide hardcoded config for use across scripts.

Latest changes: 14/08/26:
- Added STOCKFISH_PATH
"""

####################
# Data Folders
####################

DATA_DIR = '/content/drive/MyDrive/drive_diss/data'
RAW_DUMP_DIR = '/content/drive/MyDrive/drive_diss/data/raw_dumps'
GAME_DIR = '/content/drive/MyDrive/drive_diss/data/game_data'
POS_DIR = '/content/drive/MyDrive/drive_diss/data/pos_data'
IDS_DIR = '/content/drive/MyDrive/drive_diss/data/pos_data/ids'
TRAIN_DIR = '/content/drive/MyDrive/drive_diss/data/pos_data/train'
VAL_DIR = '/content/drive/MyDrive/drive_diss/data/pos_data/val'
TEST_DIR = '/content/drive/MyDrive/drive_diss/data/pos_data/test'
RESULTS_DIR = '/content/drive/MyDrive/drive_diss/data/results'

####################
# Local Disk - Colab session 
####################

LOCAL_SPLITS_DIR = '/content/prepared_splits'

####################
# Checkpoints
####################

CHECKPOINTS_DIR = '/content/drive/MyDrive/drive_diss/checkpoints'

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