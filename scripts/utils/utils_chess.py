"""
utils_chess.py
Chess-specific helpers: FEN parsing, board encoding, material/phase computation.

Latest changes: 24/08/26:
- Added material and title-strength helpers
"""

import numpy as np
import pandas as pd
import torch
import chess
from dataclasses import dataclass

####################
# CONSTANTS
####################

# Game phase
PAWN_PHASE = 0
KNIGHT_PHASE = 1
BISHOP_PHASE = 1
ROOK_PHASE = 2
QUEEN_PHASE = 4

TOTAL_PHASE = (PAWN_PHASE * 16 + KNIGHT_PHASE * 4 + BISHOP_PHASE * 4
               + ROOK_PHASE * 4 + QUEEN_PHASE * 2)

OPENING_MAX_PHASE = 64
MIDDLEGAME_MAX_PHASE = 192

# Material values (pawns included, kings excluded)
PAWN_VALUE = 1
KNIGHT_VALUE = 3
BISHOP_VALUE = 3
ROOK_VALUE = 5
QUEEN_VALUE = 9

PIECE_VALUES = {
    chess.PAWN: PAWN_VALUE,
    chess.KNIGHT: KNIGHT_VALUE,
    chess.BISHOP: BISHOP_VALUE,
    chess.ROOK: ROOK_VALUE,
    chess.QUEEN: QUEEN_VALUE,
}

# Fen handling
FEN_PAD = '.'
BOARD_SEQ_LEN = 76
FEN_CHARS = sorted(set('PNBRQKpnbrqk.wabcdefgh0123456789'))
FEN_VOCAB = {'<pad>': 0, **{ch: i + 1 for i, ch in enumerate(FEN_CHARS)}}

# Elo gap
GAP_BIN_WIDTH = 10

# Result encoding
RESULT_TO_CLASS = {0.0: 0, 0.5: 1, 1.0: 2} 
RESULT_CLASS_NAMES = ['loss', 'draw', 'win']

# Title encoding. Fixed vocabulary matching Lichess's title set (lichess.org/help/master,
# lichess.org/qa/4451), rather than fit from train data, since it's a small, effectively
# closed set. 'no_title' matches the label run_pos_storage.py substitutes for the raw 'None'
# sentinel; 'unk' is a safety net for any value outside this vocabulary.
TITLE_TO_IDX = {
    'no_title': 0,
    'GM': 1,
    'WGM': 2,
    'IM': 3,
    'WIM': 4,
    'FM': 5,
    'WFM': 6,
    'CM': 7,
    'WCM': 8,
    'NM': 9,
    'WNM': 10,
    'LM': 11,
    'BOT': 12,
    'unk': 13,
}
TITLE_VOCAB_SIZE = len(TITLE_TO_IDX)

# Title strength, within-track only -- open and women's titles are separate scales and are
# never compared directly. Higher = stronger. LM has no women's-track counterpart.
OPEN_TITLE_STRENGTH = {'GM': 6, 'IM': 5, 'FM': 4, 'CM': 3, 'NM': 2, 'LM': 1}
WOMENS_TITLE_STRENGTH = {'WGM': 6, 'WIM': 5, 'WFM': 4, 'WCM': 3, 'WNM': 2}

####################
# FUNCTIONS
####################

# (a) FEN CONVERSIONS INTO MODEL INPUT - MOVER PERSPECTIVE, NOT WHITE

def _mover_perspective_board(fen: str) -> tuple[chess.Board, bool]:
    """Parses a FEN and mirrors the board so the side to move is always white."""
    board = chess.Board(fen)
    mover_is_white = board.turn == chess.WHITE
    if not mover_is_white:
        board = board.mirror()
    return board, mover_is_white


def fen_to_tensor(fen: str) -> torch.Tensor:
    """Converts a FEN string into an (18, 8, 8) mover-perspective input tensor."""
    board, mover_is_white = _mover_perspective_board(fen)

    arr = np.zeros((18, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        channel = (piece.piece_type - 1) + (0 if piece.color else 6)
        arr[channel, square >> 3, square & 7] = 1.0

    if mover_is_white:
        arr[12] = 1.0
    if board.has_kingside_castling_rights(chess.WHITE):
        arr[13] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        arr[14] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        arr[15] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        arr[16] = 1.0

    if board.ep_square is not None:
        arr[17, board.ep_square >> 3, board.ep_square & 7] = 1.0

    return torch.from_numpy(arr)


def fen_to_char_string(fen: str) -> str:
    """Expands a mover-perspective FEN into a fixed-length (76-char) board+metadata character string."""
    board, _ = _mover_perspective_board(fen)
    board_fen, active, castling, ep, halfmove, fullmove = board.fen().split(' ')

    board_chars = []
    for char in board_fen:
        if char == '/':
            continue
        elif char.isdigit():
            board_chars.extend([FEN_PAD] * int(char))
        else:
            board_chars.append(char)
    assert len(board_chars) == 64, f'Expected 64 board chars, got {len(board_chars)}'

    castling = ''.join(c for c in castling if c in 'KQkq').ljust(4, FEN_PAD)
    ep = FEN_PAD * 2 if ep == '-' else ep.ljust(2, FEN_PAD)
    halfmove = halfmove.rjust(2, FEN_PAD)
    fullmove = fullmove.rjust(3, FEN_PAD)

    return ''.join(board_chars) + active + castling + ep + halfmove + fullmove


def fen_to_token_ids(fen: str) -> torch.Tensor:
    """Converts a FEN into a LongTensor of FEN_VOCAB indices."""
    ids = [FEN_VOCAB.get(ch, 0) for ch in fen_to_char_string(fen)]
    return torch.tensor(ids, dtype=torch.long)


def fen_to_legal_dest(fen: str) -> torch.Tensor:
    """Converts a FEN into a (64,) mover-perspective legal-move-destination multi-hot tensor."""
    board, _ = _mover_perspective_board(fen)
    arr = np.zeros(64, dtype=np.float32)
    for move in board.legal_moves:
        arr[move.to_square] = 1.0
    return torch.from_numpy(arr)


def fen_to_attacked_squares(fen: str) -> torch.Tensor:
    """Converts a FEN into a (2, 64) mover-perspective attacked-squares tensor: row 0 mover, row 1 opponent."""
    board, _ = _mover_perspective_board(fen)
    arr = np.zeros((2, 64), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row = 0 if piece.color == chess.WHITE else 1
        for attacked in board.attacks(square):
            arr[row, attacked] = 1.0
    return torch.from_numpy(arr)


# (b) COMPUTE GAME PHASE & MATERIAL FROM FEN

def game_phase(fen: str) -> int:
    """Computes a material-based game phase score from a FEN (0 = full material, 256 = bare endgame)."""
    placement = fen.split(' ', 1)[0]
    minor_major = (
        (placement.count('N') + placement.count('n')) * KNIGHT_PHASE
        + (placement.count('B') + placement.count('b')) * BISHOP_PHASE
        + (placement.count('R') + placement.count('r')) * ROOK_PHASE
        + (placement.count('Q') + placement.count('q')) * QUEEN_PHASE
    )
    phase = max(0, TOTAL_PHASE - minor_major)
    return (phase * 256 + TOTAL_PHASE // 2) // TOTAL_PHASE


def phase_label(phase: int) -> str:
    """Buckets a game phase score into opening, middlegame, or endgame."""
    if phase <= OPENING_MAX_PHASE:
        return 'opening'
    elif phase <= MIDDLEGAME_MAX_PHASE:
        return 'middlegame'
    return 'endgame'


def total_material(fen: str) -> int:
    """Sums standard piece values (pawns included, kings excluded) for both sides from a FEN."""
    board = chess.Board(fen)
    return sum(PIECE_VALUES.get(piece.piece_type, 0) for piece in board.piece_map().values())


def material_diff(fen: str) -> int:
    """Mover-perspective signed material difference (mover minus opponent), standard values, pawns included."""
    board, _ = _mover_perspective_board(fen)
    diff = 0
    for piece in board.piece_map().values():
        value = PIECE_VALUES.get(piece.piece_type, 0)
        diff += value if piece.color == chess.WHITE else -value
    return diff

# (c) GENERAL CONVERSION FROM WHITE-MOVER PERSPECTIVE

def to_mover_perspective(df: pd.DataFrame, column: str, mover_is_white_col: str = 'mover_is_white') -> pd.DataFrame:
    """Flips a white-perspective column's sign to mover perspective based on mover_is_white."""
    df = df.copy()
    is_white = df[mover_is_white_col].to_numpy()
    df[column] = np.where(is_white, df[column].to_numpy(), -df[column].to_numpy())
    return df


def to_mover_opponent_perspective(df: pd.DataFrame, white_col: str, black_col: str,
                                   mover_is_white_col: str = 'mover_is_white') -> pd.DataFrame:
    """Replaces a white/black column pair with perspective-flipped mover/opponent columns."""
    mover_col = white_col.replace('white', 'mover')
    opponent_col = black_col.replace('black', 'opponent')

    df = df.copy()
    is_white = df[mover_is_white_col].to_numpy()
    white_vals = df[white_col].to_numpy()
    black_vals = df[black_col].to_numpy()

    df[mover_col] = np.where(is_white, white_vals, black_vals)
    df[opponent_col] = np.where(is_white, black_vals, white_vals)

    return df.drop(columns=[white_col, black_col])


# (d) ELO-BINNING PROCESSES

@dataclass(frozen=True)
class EloBinConfig:
    """Elo bin edges: lower, upper, step, and whether tails are open."""
    lower: int = 800
    upper: int = 2200
    step: int = 200
    tails: bool = True

ELO_BINS = EloBinConfig()


def elo_bin_edges(cfg: EloBinConfig = ELO_BINS) -> list[float]:
    """Returns bin edges from cfg, with open tails if cfg.tails."""
    edges = list(range(cfg.lower, cfg.upper + 1, cfg.step))
    if cfg.tails:
        edges = [-np.inf] + edges + [np.inf]
    return edges


def elo_bin_labels(edges: list[float]) -> list[str]:
    """Returns a display label per bin, e.g. '<800', '800-999', '>=2200'."""
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if lo == -np.inf:
            labels.append(f'<{int(hi)}')
        elif hi == np.inf:
            labels.append(f'>={int(lo)}')
        else:
            labels.append(f'{int(lo)}-{int(hi) - 1}')
    return labels


def elo_bin_by_mover(df: pd.DataFrame, cfg: EloBinConfig = ELO_BINS, method: str = 'mean') -> tuple[pd.DataFrame, list[float]]:
    """Bins games by elo (mean of mover/opponent, or mover-only) into cfg's edges."""
    if method not in ('mean', 'mover'):
        raise ValueError(f"method must be 'mean' or 'mover', got {method!r}")

    edges = elo_bin_edges(cfg)
    df = df.copy()
    elo = (df['mover_elo'] + df['opponent_elo']) / 2 if method == 'mean' else df['mover_elo']

    elo_bin = pd.cut(elo, bins=edges, labels=False, right=False)
    if elo_bin.isna().any():
        raise ValueError(f'{elo_bin.isna().sum()} rows handled incorrectly - null.')

    df['elo_bin'] = elo_bin.astype(int)
    return df, edges


# (e) ELO-GAP RESTRICTION AND GAME-ID FILTERING

def _dedup_to_game_level(df: pd.DataFrame, game_id_col: str = 'game_id') -> pd.DataFrame:
    """Collapses position-level rows to one row per game_id."""
    n_before = len(df)
    game_df = df.drop_duplicates(subset=game_id_col).copy()
    if len(game_df) != n_before:
        print(f'  [_dedup_to_game_level] collapsed {n_before:,} rows -> {len(game_df):,} games')
    return game_df


def _res_better(df: pd.DataFrame) -> pd.DataFrame:
    """Adds res_better: the result from the higher-elo player's perspective."""
    df = df.copy()
    df['res_better'] = np.where(df['mover_elo'] >= df['opponent_elo'],
                                 df['mover_result'], 1 - df['mover_result'])
    return df


def _find_threshold_crossing(bin_centres: list[float], values: list, target: float) -> float | None:
    """Interpolates the gap value at which values first reaches target."""
    pairs = [(c, v) for c, v in zip(bin_centres, values) if v is not None]
    if not pairs:
        return None
    centres, vals = zip(*pairs)
    above = [v >= target for v in vals]
    if not any(above) or all(above):
        return None
    idx = above.index(True)
    if idx == 0:
        return centres[0]
    x0, x1 = centres[idx - 1], centres[idx]
    y0, y1 = vals[idx - 1], vals[idx]
    return x0 + (target - y0) * (x1 - x0) / (y1 - y0)


def _compute_gap_threshold(group: pd.DataFrame, target_score: float, gap_bin_width: int) -> float | None:
    """Fits the win-rate-by-gap curve for one bracket, finds where it crosses target_score."""
    nonzero = group[group['elo_gap'] > 0]
    if nonzero.empty:
        return None
    gap_bin_idx = np.ceil(nonzero['elo_gap'] / gap_bin_width).astype(int) - 1
    stats = nonzero.groupby(gap_bin_idx)['res_better'].mean()
    max_idx = stats.index.max()
    bin_centres = [idx * gap_bin_width + gap_bin_width / 2 for idx in range(max_idx + 1)]
    expected = [stats.get(idx) for idx in range(max_idx + 1)]
    threshold = _find_threshold_crossing(bin_centres, expected, target_score)
    return round(threshold) if threshold is not None else None


def fit_elo_gap_thresholds(df_train: pd.DataFrame, target_score: float, cfg: EloBinConfig = ELO_BINS,
                            game_id_col: str = 'game_id', gap_bin_width: int = GAP_BIN_WIDTH,
                            on_missing: str = 'keep_all') -> dict:
    """Fits per-elo-bracket gap thresholds at which the higher-elo player reaches target_score."""
    if on_missing not in ('keep_all', 'drop_all'):
        raise ValueError(f"on_missing must be 'keep_all' or 'drop_all', got {on_missing!r}")

    game_df = _dedup_to_game_level(df_train, game_id_col)
    binned, edges = elo_bin_by_mover(game_df, cfg, method='mean')
    binned['elo_gap'] = (binned['mover_elo'] - binned['opponent_elo']).abs()
    binned = _res_better(binned)
    labels = elo_bin_labels(edges)
    n_bins = len(edges) - 1

    global_threshold = _compute_gap_threshold(binned, target_score, gap_bin_width)
    print(f'Global elo-gap threshold (all brackets pooled): {global_threshold}')

    fallback_value = np.inf if on_missing == 'keep_all' else -1

    thresholds = {}
    used_global_fallback = []
    used_hardcoded_fallback = []

    for i in range(n_bins):
        group = binned[binned['elo_bin'] == i]
        n_games = len(group)
        t = _compute_gap_threshold(group, target_score, gap_bin_width)
        if t is None:
            if global_threshold is not None:
                t = global_threshold
                used_global_fallback.append((i, labels[i], n_games))
            else:
                t = fallback_value
                used_hardcoded_fallback.append((i, labels[i], n_games))
        thresholds[i] = t

    if used_global_fallback:
        print(f'\n{len(used_global_fallback)} bracket(s) had no own threshold; fell back to global ({global_threshold}):')
        for i, label, n in used_global_fallback:
            print(f'  bin {i} ({label}): n={n:,} games in bracket')

    if used_hardcoded_fallback:
        print(f"\n{len(used_hardcoded_fallback)} bracket(s) had NO own threshold AND no usable global fallback; "
              f"used on_missing='{on_missing}' (threshold={fallback_value}):")
        for i, label, n in used_hardcoded_fallback:
            print(f'  bin {i} ({label}): n={n:,} games in bracket')

    print('\nPer-bracket thresholds:')
    for i in range(n_bins):
        print(f'  {labels[i]:>12}: gap <= {thresholds[i]}')

    return {
        'thresholds': thresholds,
        'edges': edges,
        'labels': labels,
        'config': dict(target_score=target_score, cfg=cfg, game_id_col=game_id_col, gap_bin_width=gap_bin_width),
    }


def apply_elo_gap_thresholds(df: pd.DataFrame, fit_result: dict, verbose: bool = True) -> list:
    """Returns the game_ids passing the fitted elo-gap thresholds."""
    fit_cfg = fit_result['config']
    thresholds = fit_result['thresholds']
    game_id_col = fit_cfg['game_id_col']

    game_df = _dedup_to_game_level(df, game_id_col)
    binned, _ = elo_bin_by_mover(game_df, fit_cfg['cfg'], method='mean')
    binned['elo_gap'] = (binned['mover_elo'] - binned['opponent_elo']).abs()

    row_thresholds = binned['elo_bin'].map(thresholds)
    keep_mask = binned['elo_gap'] <= row_thresholds
    kept_game_ids = binned.loc[keep_mask, game_id_col].tolist()

    if verbose:
        n_games_in = len(game_df)
        n_games_over_gap = int((~keep_mask).sum())
        n_games_kept = len(kept_game_ids)

        n_rows_in = len(df)
        n_rows_kept = int(df[game_id_col].isin(kept_game_ids).sum())

        print(f'apply_elo_gap_thresholds (game level): {n_games_in:,} games in -> '
              f'{n_games_over_gap:,} over gap threshold, {n_games_kept:,} games kept')
        print(f'  input rows: {n_rows_in:,} -> {n_rows_kept:,} rows would be kept '
              f'({n_rows_in - n_rows_kept:,} dropped) if filtered by this game_id list')

    return kept_game_ids


def filter_by_game_ids(df: pd.DataFrame, game_ids: list, game_id_col: str = 'game_id') -> pd.DataFrame:
    """Filters df to rows whose game_id_col is in game_ids."""
    return df[df[game_id_col].isin(game_ids)].reset_index(drop=True)


# (f) RESULT ENCODING

def encode_result_continuous(mover_result: pd.Series) -> torch.Tensor:
    """Returns mover_result as a float32 tensor."""
    return torch.tensor(mover_result.to_numpy(), dtype=torch.float32)

def encode_result_class(mover_result: pd.Series) -> torch.Tensor:
    """Maps mover_result (0/0.5/1) to a class index (loss/draw/win) tensor."""
    return torch.tensor(mover_result.map(RESULT_TO_CLASS).to_numpy(), dtype=torch.long)

def decode_result_class(class_idx: np.ndarray | torch.Tensor | list[int]) -> list[str]:
    """Maps class indices (0/1/2) back to result names (loss/draw/win)."""
    return [RESULT_CLASS_NAMES[int(c)] for c in class_idx]


# (g) TITLE ENCODING

def encode_title_idx(title: pd.Series) -> np.ndarray:
    """Maps a title Series to TITLE_TO_IDX indices as a plain int64 array."""
    return title.astype('object').map(TITLE_TO_IDX).fillna(TITLE_TO_IDX['unk']).to_numpy(dtype='int64')


def title_track(title: str) -> str | None:
    """Returns 'open', 'womens', or None (untitled/BOT/unrecognized) for a title string."""
    if title in OPEN_TITLE_STRENGTH:
        return 'open'
    if title in WOMENS_TITLE_STRENGTH:
        return 'womens'
    return None


def title_strength(title: str) -> int | None:
    """Returns the within-track strength rank for title, or None if untitled/unrecognized."""
    if title in OPEN_TITLE_STRENGTH:
        return OPEN_TITLE_STRENGTH[title]
    return WOMENS_TITLE_STRENGTH.get(title)