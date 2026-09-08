"""
utils_overleaf.py
Path resolution for saving figures/tables into subject-specific Overleaf subfolders.

Latest changes: 08/09/26:
- Fixed error in push_to_overleaf v2
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
    untracked = subprocess.run(['git', '-C', repo_dir, 'ls-files', '--others', '--exclude-standard', *paths],
                                capture_output=True, text=True, check=True).stdout.splitlines()

    cached_bytes = {}
    for rel_path in untracked:
        abs_path = os.path.join(repo_dir, rel_path)
        with open(abs_path, 'rb') as f:
            cached_bytes[abs_path] = f.read()
        os.remove(abs_path)

    pull = subprocess.run(['git', '-C', repo_dir, 'pull'], capture_output=True, text=True)
    if pull.returncode != 0:
        for abs_path, content in cached_bytes.items():
            with open(abs_path, 'wb') as f:
                f.write(content)
        raise RuntimeError(f'push_to_overleaf: git pull failed:\n{pull.stderr}')

    for abs_path, content in cached_bytes.items():
        with open(abs_path, 'wb') as f:
            f.write(content)

    subprocess.run(['git', '-C', repo_dir, 'add', *paths], check=True)

    status = subprocess.run(['git', '-C', repo_dir, 'status', '--porcelain', *paths],
                             capture_output=True, text=True, check=True)
    if not status.stdout.strip():
        print('No changes to push.')
        return

    subprocess.run(['git', '-C', repo_dir, 'commit', '-m', message], check=True)
    push = subprocess.run(['git', '-C', repo_dir, 'push'], capture_output=True, text=True)
    if push.returncode != 0:
        raise RuntimeError(f'push_to_overleaf: git push failed:\n{push.stderr}')