"""
run_download_dumps.py
Downloads one or more monthly lichess .pgn.zst dumps to Drive.

Latest changes: 14/08/26:
- Initial commit

Run:
    !python run_download_dumps.py --date 2024-01 2024-02 --out-dir /content/drive/MyDrive/drive_diss/data/raw_dumps
CLI:
    --date       One or more dump months, YYYY-MM. Required.
    --out-dir    Folder to save into. Required.
"""

import argparse
import os

import requests
from tqdm.auto import tqdm

####################
# CONSTANTS
####################

BASE_URL = 'https://database.lichess.org/standard/lichess_db_standard_rated_{date}.pgn.zst'

####################
# FUNCTIONS
####################

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses --date and --out-dir CLI arguments."""
    parser = argparse.ArgumentParser(description='Download lichess monthly dumps to Drive.')
    parser.add_argument('--date', required=True, nargs='+', help='One or more dump months, YYYY-MM.')
    parser.add_argument('--out-dir', required=True, help='Folder to save into.')
    return parser.parse_args(argv)


def download_dump(date: str, out_dir: str) -> None:
    """Streams one month's .pgn.zst dump from lichess to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    url = BASE_URL.format(date=date)
    local_path = os.path.join(out_dir, f'lichess_{date}.pgn.zst')
    print(f'Source: {url}')
    print(f'Target: {local_path}')
    response = requests.get(url, stream=True)
    response.raise_for_status()
    remote_size = int(response.headers.get('content-length', 0))
    bytes_written = 0
    with open(local_path, 'wb') as f, tqdm(total=remote_size or None, unit='B', unit_scale=True, desc=date) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bytes_written += len(chunk)
            bar.update(len(chunk))
    response.close()
    print(f'Done: {bytes_written / 1024**3:.2f} GB -> {local_path}')


def main(argv: list[str] | None = None) -> None:
    """Downloads every requested month's dump to args.out_dir."""
    args = parse_args(argv)
    for date in args.date:
        download_dump(date, args.out_dir)


if __name__ == '__main__':
    main()