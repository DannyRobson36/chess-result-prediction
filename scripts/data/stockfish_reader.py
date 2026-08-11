"""
stockfish_reader.py
Stockfish win/loss-only evaluation of positions in an input CSV, output directly in
model-eval-ready form: game_id, fen, eval_{depth}, prob_win, prob_draw, prob_loss,
predicted_class.

CLI (all required, no defaults):
    --input-path    Path to the input CSV to evaluate. Must contain game_id, fen.
    --output-path   Full path (including filename) to write the output CSV to.
    --depth         Single Stockfish search depth to evaluate at.

Run:
    !python stockfish_reader.py --input-path /path/to/val_pos.csv --output-path /path/to/val_predictions_stockfish_7.csv --depth 7

Latest changes: 11/08/26:
- Initial commit
"""

import ast
import importlib.util
import subprocess
import sys

import argparse
import csv
import os
import signal
import multiprocessing as mp
from types import FrameType

import numpy as np
import pandas as pd
from tqdm import tqdm

import chess
import chess.engine

####################
# CONSTANTS
####################

MATE_SCORE_MOVER = 100_000
WIN_PCT_CONST = 0.00368208
HASH_MB_PER_ENGINE = 16
ATTEMPT_TIMEOUT_SECONDS = 120
ENGINE_PATH = '/usr/games/stockfish'
REQUIRED_COLUMNS = ['game_id', 'fen']
NUM_WORKERS = os.cpu_count() or 1

####################
# FUNCTIONS
####################

# (a) VALIDATION & SETUP

def _validate_input_columns(fieldnames: list[str] | None, input_path: str) -> None:
    """Exits if any REQUIRED_COLUMNS are missing from the input CSV's header."""
    missing = [c for c in REQUIRED_COLUMNS if c not in (fieldnames or [])]
    if missing:
        sys.exit(f'{input_path} is missing required column(s): {missing}. '
                 f'Required: {REQUIRED_COLUMNS}')

def ensure_stockfish_installed(engine_path: str) -> None:
    """Installs Stockfish via apt-get if not already present at engine_path."""
    if os.path.exists(engine_path):
        return
    print('Stockfish not found -- installing via apt-get...')
    install = subprocess.run(['apt-get', 'install', '-y', 'stockfish'], capture_output=True, text=True)
    if install.returncode != 0 or not os.path.exists(engine_path):
        sys.exit(f'Stockfish install failed or binary not found at {engine_path}.\n{install.stderr}')
    print('Stockfish installed successfully.')

# (b) LOADING

def load_positions(input_path: str) -> list[dict[str, str]]:
    """Loads every row of the input CSV, in order, keeping only REQUIRED_COLUMNS."""
    with open(input_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        _validate_input_columns(reader.fieldnames, input_path)
        rows = [{col: row[col] for col in REQUIRED_COLUMNS} for row in reader]
    return rows

# (c) WORKER SETUP

def _start_engine() -> chess.engine.SimpleEngine:
    """Starts a single-threaded Stockfish engine process."""
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    engine.configure({'Threads': 1, 'Hash': HASH_MB_PER_ENGINE})
    return engine

def _init_worker() -> None:
    """Starts this worker process's persistent engine as a module-global."""
    global _engine
    _engine = _start_engine()

def _restart_engine() -> None:
    """Closes (or kills) and restarts this worker's engine."""
    global _engine
    try:
        _engine.close()
    except Exception:
        try:
            _engine.transport.kill()
        except Exception:
            pass
    _engine = _start_engine()

def _alarm_handler(signum: int, frame: FrameType | None) -> None:
    """Raises _TaskTimeout when the per-attempt SIGALRM fires."""
    raise _TaskTimeout()

def _analyse_clean(board: chess.Board, depth: int) -> chess.engine.InfoDict:
    """Clears the engine's hash table, then analyses board to depth."""
    _engine.configure({'Clear Hash': None})
    return _engine.analyse(board, chess.engine.Limit(depth=depth))

# (d) PER-POSITION EVALUATION

def _terminal_eval(board: chess.Board) -> int:
    """Returns the mover-perspective eval for a terminal position: mate or draw."""
    if board.is_checkmate():
        return -MATE_SCORE_MOVER
    return 0

def _process_one_inner(fen: str, depth: int) -> tuple[int, bool]:
    """Evaluates one position at one depth, skipping the engine for terminal positions."""
    board = chess.Board(fen)
    mover_is_white = board.turn

    if board.is_game_over():
        return _terminal_eval(board), True

    info = _analyse_clean(board, depth)
    white_score = info['score'].white()

    if white_score.is_mate():
        mate_for_white = white_score.mate() > 0
        mate_for_mover = (mate_for_white == mover_is_white)
        eval_val = MATE_SCORE_MOVER if mate_for_mover else -MATE_SCORE_MOVER
    else:
        raw_cp = white_score.score()
        eval_val = raw_cp if mover_is_white else -raw_cp

    return eval_val, False

def _process_one(args: tuple[str, int]) -> dict[str, int | bool]:
    """Evaluates one position, stepping depth down by 1 per timeout or error until depth 1."""
    fen, target_depth = args
    for depth in range(target_depth, 0, -1):
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(ATTEMPT_TIMEOUT_SECONDS)
        try:
            eval_val, is_terminal = _process_one_inner(fen, depth)
            signal.alarm(0)
            fallback = (not is_terminal) and (depth != target_depth)
            return {'eval': eval_val, 'is_terminal': is_terminal, 'fallback': fallback}
        except _TaskTimeout:
            signal.alarm(0)
            _restart_engine()
        except Exception:
            signal.alarm(0)
            _restart_engine()
    raise RuntimeError(
        f'Stockfish failed to evaluate this position even at depth 1 '
        f'(every depth from {target_depth} down to 1 timed out or errored): {fen}'
    )

# (e) PROBABILITY CONVERSION

def eval_to_win_prob(cp: np.ndarray | float) -> np.ndarray:
    """Maps mover-perspective centipawns to a win probability via a fixed logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-WIN_PCT_CONST * np.asarray(cp, dtype='float64')))

# (f) CLI

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses and validates the --input-path, --output-path, --depth CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Stockfish win/loss evaluation of positions in an input CSV, '
                     'output directly in utils_eval-ready form.'
    )
    parser.add_argument('--input-path', required=True,
                         help='Path to the input CSV to evaluate. Must contain game_id, fen.')
    parser.add_argument('--output-path', required=True,
                         help='Full path (including filename) to write the output CSV to.')
    parser.add_argument('--depth', type=int, required=True,
                         help='Single Stockfish search depth to evaluate at.')
    args = parser.parse_args(argv)
    if args.depth <= 0:
        parser.error('--depth must be > 0.')
    return args

# (g) MAIN

def main(argv: list[str] | None = None) -> None:
    """Runs the full pipeline: load positions, evaluate with Stockfish, write predictions CSV."""
    args = parse_args(argv)
    depth = args.depth
    input_path = args.input_path
    output_path = args.output_path

    if not os.path.exists(input_path):
        sys.exit(f'No input CSV found at {input_path}.')

    ensure_stockfish_installed(ENGINE_PATH)

    print(f'Detected {NUM_WORKERS} CPU core(s) available -- using all of them as workers.')
    print(f'Mate evals stored as +-{MATE_SCORE_MOVER} in eval_{depth} (mover-perspective).')
    print(f'Per-attempt timeout: {ATTEMPT_TIMEOUT_SECONDS}s, falling back one depth at a time, '
          f'no retry at the same depth, hard error if depth 1 also fails.')

    print(f'Loading positions from: {input_path}')
    rows = load_positions(input_path)
    print(f'Loaded {len(rows):,} position(s) -- every row will get an evaluation.')

    tasks = [(row['fen'], depth) for row in rows]

    print(f'Analysing {len(tasks):,} positions with {NUM_WORKERS} worker(s) at depth {depth}...')

    results = []
    with mp.Pool(processes=NUM_WORKERS, initializer=_init_worker) as pool:
        for res in tqdm(pool.imap(_process_one, tasks), total=len(tasks), desc='positions', unit='pos'):
            results.append(res)

    assert len(results) == len(rows), (
        f'Internal error: got {len(results)} results for {len(rows)} input rows -- '
        f'row alignment broke somewhere.'
    )

    evals = np.array([r['eval'] for r in results], dtype='float64')
    n_terminal = sum(r['is_terminal'] for r in results)
    n_fallback = sum(r['fallback'] for r in results)

    prob_win = eval_to_win_prob(evals)
    prob_loss = 1.0 - prob_win
    prob_draw = np.zeros(len(results), dtype='float64')
    predicted_class = np.where(prob_win >= 0.5, 'win', 'loss')

    df_out = pd.DataFrame({
        'game_id': [row['game_id'] for row in rows],
        'fen': [row['fen'] for row in rows],
        f'eval_{depth}': evals,
        'prob_win': prob_win,
        'prob_draw': prob_draw,
        'prob_loss': prob_loss,
        'predicted_class': predicted_class,
    })

    assert len(df_out) == len(rows), 'Output row count does not match input row count.'

    print(f'There were {n_terminal} terminal position(s) (expected 0).')
    print(f'There were {n_fallback} position(s) evaluated at a lower depth than requested.')

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f'Saved {len(df_out):,} rows to {output_path}')

####################
# CLASSES
####################

class _TaskTimeout(Exception):
    """Raised internally when a single Stockfish analysis attempt exceeds its timeout."""
    pass

####################
# ENTRY POINT
####################
if __name__ == '__main__':
    main()