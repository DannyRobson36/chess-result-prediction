"""
utils_setup.py
Helpers for notebook startup: reproducibility seeding, requirements
installation, and runtime info printing.

Latest changes: 17/09/26:
- Added set_seed
"""

import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import psutil
import torch

####################
# CONSTANTS
####################

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = REPO_ROOT / 'requirements.txt'

####################
# FUNCTIONS
####################

# (a) REPRODUCIBILITY

def set_seed(seed: int) -> None:
    """Seeds python, numpy, and torch (cpu + cuda)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# (b) PACKAGE INSTALLATION

def pull_reqs() -> None:
    """Installs packages listed in requirements.txt via pip."""
    print()
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-r', str(REQUIREMENTS), '-q'],
        check=True,
    )
    print('Packages installed.')

# (c) RUNTIME INFO

def print_runtime_info() -> None:
    """Prints Python, PyTorch, device, CPU core count, and RAM tier."""
    print()
    print('=' * 40)
    print('Runtime information')
    print('=' * 40)

    print(f'Python       : {platform.python_version()}')
    print(f'PyTorch      : {torch.__version__}')

    if torch.cuda.is_available():
        print('Device       : GPU')
        print(f'GPU          : {torch.cuda.get_device_name(0)}')
        print(f'CUDA         : {torch.version.cuda}')
    else:
        print('Device       : CPU')

    print(f'CPU cores    : {os.cpu_count()}')

    ram = psutil.virtual_memory().total / (1024 ** 3)
    print(f'System RAM   : {ram:.1f} GB')

    if ram >= 80:
        print('RAM tier     : Very High RAM')
    elif ram >= 45:
        print('RAM tier     : High RAM')
    else:
        print('RAM tier     : Standard RAM')

    print('=' * 40)
    print()