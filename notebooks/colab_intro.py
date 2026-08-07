import os
from google.colab import drive, userdata

if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

%cd '/content/drive/MyDrive/drive_diss/crp_notebooks'
from repo_cloner import setup_repo
REPO_ROOT = setup_repo()

%cd {REPO_ROOT}
from scripts.config import DATA_DIR, REPO_SCRIPTS_DIR
from scripts.utils.utils_setup import pull_reqs, print_runtime_info
pull_reqs()
print_runtime_info()

%cd {REPO_SCRIPTS_DIR}