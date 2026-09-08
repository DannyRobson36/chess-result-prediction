"""
utils_overleaf.py
Path resolution for saving figures/tables into subject-specific Overleaf subfolders.

Latest changes: 08/09/26:
- Initial commit
"""

import os
import subprocess

####################
# FUNCTIONS
####################

# (a) PATH RESOLUTION

def resolve_output_path(base_dir: str, subfolder: str, filename: str) -> str:
    """Returns the full path for filename under base_dir/subfolder, creating the subfolder if needed."""
    out_dir = os.path.join(base_dir, subfolder)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)


# (b) OVERLEAF SYNC

def push_to_overleaf(repo_dir: str, paths: list[str], message: str) -> None:
    """Pulls repo_dir, stages paths, commits with message, and pushes to the Overleaf remote.
    No-ops with a printed message if paths have no staged changes."""
    subprocess.run(['git', '-C', repo_dir, 'pull'], check=True)
    subprocess.run(['git', '-C', repo_dir, 'add', *paths], check=True)

    status = subprocess.run(['git', '-C', repo_dir, 'status', '--porcelain', *paths],
                             capture_output=True, text=True, check=True)
    if not status.stdout.strip():
        print('No changes to push.')
        return

    subprocess.run(['git', '-C', repo_dir, 'commit', '-m', message], check=True)
    subprocess.run(['git', '-C', repo_dir, 'push'], check=True)