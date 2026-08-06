# utils_setup.py
"""
Helpers for regular notebook information.

Latest changes: 06/08/26:
- Improved spacing v2
"""

import os 
import subprocess
import sys
from pathlib import Path
import platform
import psutil
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = REPO_ROOT / 'requirements.txt'

def pull_reqs():
    print()
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-r', str(REQUIREMENTS), '-q'],
        check=True,
    )

    print('Packages installed.')

def print_runtime_info():
    print()
    print('=' * 40)
    print('Runtime information')
    print('=' * 40)

    # Python
    print(f'Python       : {platform.python_version()}')

    # PyTorch
    print(f'PyTorch      : {torch.__version__}')

    # Device
    if torch.cuda.is_available():
        print('Device       : GPU')
        print(f'GPU          : {torch.cuda.get_device_name(0)}')
        print(f'CUDA         : {torch.version.cuda}')
    else:
        print('Device       : CPU')

    # CPU
    print(f'CPU cores    : {os.cpu_count()}')

    # RAM
    ram = psutil.virtual_memory().total / (1024**3)
    print(f'System RAM   : {ram:.1f} GB')

    if ram >= 80:
        print('RAM tier     : Very High RAM')
    elif ram >= 45:
        print('RAM tier     : High RAM')
    else:
        print('RAM tier     : Standard RAM')

    print('=' * 40)
    print()