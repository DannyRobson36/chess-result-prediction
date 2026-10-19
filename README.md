# Chess Result Prediction

This is the code for my MSc dissertation in Statistical Science at Oxford (2026), "Predicting Human Chess Results from Board Position and Context".

This project seeks to predict whether the player to move in a human chess game goes on to win, draw or lose, using the position on the board together with the context around it, such as both players' ratings, the time left on each clock, and how each player has been playing recently.

Chess engines like Stockfish are very good at judging who is better in a position, but they assume perfect play from that point on, and humans don't play perfectly. A position a 1200-rated player has won on paper can easily be thrown away, especially with two minutes left on the clock. So instead of measuring human games against optimal play, I trained neural networks directly on the outcomes of human games, giving them the board and the surrounding context together.

## What I found

The best model (a CNN with an attention layer, trained on 62.5 million positions) reaches 64.6% validation accuracy on the full three-way win/draw/loss task.

Stockfish and Maia-2 each give a single number per position (an evaluation or a win probability), and in that form they have no way of predicting a draw. To compare against them fairly, I also scored everything on a held-out test set of decisive games only:

| Model | Accuracy | Log loss | Brier score |
|---|---|---|---|
| **CNN-attention (this project)** | **70.94%** | **0.530** | **0.361** |
| Logistic regression (Stockfish eval + same context features) | 70.58% | 0.555 | 0.377 |
| Maia-2 (off the shelf) | 70.10% | 0.549 | 0.371 |
| Stockfish, depth 1 (Lichess win% curve) | 69.82% | 0.632 | 0.401 |

Maia-2 is primarily a move-prediction model. Its predictions here come from its value head, used as released, and it was trained on Lichess games up to November 2023, so some of its gap may come from the age of its training data rather than the model itself.

All four models sit within about 1.1 percentage points of each other. That's a small margin, but human results are clearly very noisy, and even getting that far took a lot. The best results came from the largest dataset and model I could train, and accuracy improved with every fivefold increase in training data, from 100k up to 62.5m positions. The context features show their value most clearly in specific situations:

- **Time pressure.** Only around 12% of games end on time, but in those games the CNN was right 68.9% of the time, against 58.0% for Maia-2, which has no clock information. In an ablation at 2.5m positions, removing the clock features cost over 8 points of accuracy on those games.
- **New players.** Lichess starts every new account at 1500, which turns out to overrate most newcomers, who win only about a third of their first games. With a flag for "rated exactly 1500 with no history", the CNN reached 79.8% accuracy in games involving those players, against 70.9% elsewhere.
- **Endgames.** Accuracy rises through the game for every model, but the CNN pulls furthest ahead at the end, reaching 87.4% in the endgame, 1.2 points clear of the logistic regression.

Elo itself turned out to be less useful than I expected. Once I'd removed games with large rating gaps (so that results across skill levels could be compared fairly), dropping Elo from the model barely changed accuracy, although predictions were better calibrated with it in.

I also built a pure transformer that reads the board as a sequence of FEN characters. It beat the CNN at every dataset size where both were trained (up to 2.5M positions), but it's around six times slower per epoch, so I couldn't afford to scale it further. That comparison is still open.

All accuracy figures are on Elo-bin-balanced evaluation sets, so every skill level from under 800 to over 2200 counts equally. The full results, ablations and caveats are in the dissertation.

## How it works

**Data.** Rated Rapid games (10 or 15 minute) from the [Lichess open database](https://database.lichess.org/), January to May 2026. Bots, abandoned games, games under 20 plies and games with large rating gaps are removed. The split is by date, so there's no leakage of a player's history between sets. Training uses February to April, validation the first half of May, and testing the second half. From each game I sample roughly one position per ten plies, skipping the first ten plies and any terminal positions.

**Inputs.** Every position is seen from the mover's perspective (the board is mirrored when it's black to move). Alongside the board, each model gets:

- both players' ratings and remaining clock time
- the time control (base time and whether there's an increment)
- a past-performance score over each player's last 10 games
- hours since each player's last game
- the result of the previous game if it's a rematch
- the new-player flag
- FIDE titles
- how many plies have been played

**Models.** There are two main architectures:

- **CNN-attention** (`cnn_attention`). An AlphaZero-style residual CNN encodes the board, and its output is turned into tokens that sit alongside one token per context feature in a self-attention layer. This was inspired by Maia-2, but here the ratings are ordinary feature tokens rather than being added to the attention queries.
- **Pure transformer** (`pure_transformer`). The board is expanded into a fixed 76-character FEN sequence, with one token per square, and fed through self-attention together with the same feature tokens.

Both are trained as three-way classifiers. They also have an auxiliary head predicting legal-move squares and attacked squares, which helped the models learn the board itself instead of relying purely on the noisy outcome signal.

## Repository layout

```
src/
  config.py              paths and constants (reads .env)
  data/                  download, filter, sample and size the datasets
  features/features.py   feature preparation, scaling and board encoding
  models/                architectures, configs and shared building blocks
  training/              training loop and grid search
  evaluation/            inference, Stockfish/Maia-2 baselines, comparison plots
  utils/                 chess, data, evaluation, plotting and LaTeX helpers
```

## Setup

I ran everything in Google Colab with Google Drive mounted, and the paths in `src/config.py` assume that layout (`/content/drive/MyDrive/<PROJECT_NAME>/...`). It will run elsewhere, but you'll need to edit the paths in `config.py`.

```bash
git clone https://github.com/DannyRobson36/chess-result-prediction.git
cd chess-result-prediction
pip install -r requirements.txt
cp .env.example .env
```

Then fill in `.env`. All five values need to exist for `config.py` to read, even if just as placeholders.

```
PROJECT_NAME=          # folder under MyDrive holding data, checkpoints and predictions
GITHUB_USER=
REPO_NAME=             # the folder name the repo is cloned as
STOCKFISH_PATH=        # path to a Stockfish binary
OVERLEAF_PROJECT_ID=   # only used by utils_overleaf.py
```

Maia-2 is only needed for its baseline script.

None of the data is in the repo. The pipeline downloads it from Lichess, and a large amount of disk space is required, around 28 GB per month of compressed `.pgn.zst` dumps.

## Running the pipeline

The data scripts are command-line tools and are meant to be run in order. Each one reads the previous step's output from the folders set in `config.py`.

```bash
# 1. Download the monthly Lichess dumps
python src/data/run_download_dumps.py --date 2026-01 2026-02 2026-03 2026-04 2026-05 --out-dir <RAW_DUMP_DIR>

# 2. Filter games and assign them to eda/train/val/test splits
python src/data/run_game_reader.py

# 3. Sample positions from each split
python src/data/run_pos_reader.py train
python src/data/run_pos_reader.py val
python src/data/run_pos_reader.py test

# 4. Build the final datasets (nested train sizes, balanced val/test, decisive-only variants)
python src/data/run_pos_storage.py train
python src/data/run_pos_storage.py val
python src/data/run_pos_storage.py test
```

Several side scripts are as follows, used purely for analysis:

- `run_unfiltered_reader.py --date 2026-01` makes an unfiltered pass over a month's games, used for the exploratory analysis behind the filtering choices.
- `run_history_reader.py --date 2026-01` computes full per-game player histories, used to choose the form of the past-performance feature.
- `run_single_bin_pos_storage.py` builds one training set per Elo bin, for the single-skill-level models.

Every script's full set of options is at the top of the file.

## Training

Training is done from a notebook rather than the command line. The rough flow is:

```python
from src.features.features import prepare_splits
from src.models.model_arch import build_model
from src.training.training import TrainConfig, run_training, save_checkpoint

train, val, val_wl, prep_cfg = prepare_splits(df_train, df_val, df_val_wl, ...)
model = build_model('cnn_attention', model_cfg)
result = run_training(model, 'cnn_attention', train, val, TrainConfig(...))
```

`save_checkpoint` stores the weights along with everything needed to reapply the same preprocessing at inference time. It also fits two calibration steps: temperature scaling on the validation set, and a probit fitted on the decisive-only validation set, which is used when the model is compared against the two-way baselines. For hyperparameter sweeps there's `grid_search` in `src/training/grid_search.py`.

When preparing the splits, `board_mode='boards_only'` is used for the CNN-attention model and `board_mode='tokens_only'` for the transformer, so each split only stores the board format its model reads.

## Evaluation and baselines

```bash
# Trained model
python src/evaluation/run_inference.py \
    --checkpoint-path <checkpoint.pt> \
    --input-path <test_positions.csv> \
    --output-path <test_predictions.csv> \
    --probit True --collapse-method expected_score

# Baselines
python src/evaluation/run_sf_inf.py    --input-path <test_positions.csv> --split test --depth 1
python src/evaluation/run_maia2_inf.py --input-path <test_positions.csv> --split test
```

`run_inference.py` checks whether the input contains draws. If it does, it outputs the model's own calibrated win/draw/loss probabilities. If it's a decisive-only file, it collapses those to a single win probability. Stockfish and Maia-2 are used exactly as they come, with no fitting to this data.

Any comparison that includes Stockfish or Maia-2 should use the decisive-only (`_wl`) files, since their single-value outputs give no draw prediction.

`compare_predictions.py` takes a folder of prediction files and produces the comparison plots (accuracy by Elo bin, by clock time, by game phase, ROC and calibration curves). `training_diagnostics.py` covers training curves and per-bin metrics for a single run.

## Registered architectures

| Name | Output | Notes |
|---|---|---|
| `cnn_attention` | 3-way classification | The CNN-attention model from the dissertation |
| `pure_transformer` | 3-way classification | The FEN-token transformer from the dissertation |
| `log_reg_baseline` | 3-way classification | The logistic regression baseline |
| `maia2_value_replica` | regression | A rough replica of Maia-2's value head, from early experiments |
| `maia2_value_board` | regression | The replica without Elo, from early experiments |

The last two aren't used in the final results.

## Limitations

The dissertation covers these in more detail.

- **Compute.** Compute was limited to a single Colab Pro+ subscription. Most results are from a single seed, the hyperparameters for the two largest dataset sizes were extrapolated from smaller grid searches, and Stockfish only ran at depth 1.
- **Data.** Everything is Lichess Rapid. I haven't tested other time controls or over-the-board games.
- **Draws.** Even the best model only correctly identifies about 16% of drawn games, so most of its accuracy comes from decisive results.
- **CNN vs transformer.** The two architectures weren't matched on parameter count, and the transformer was only trained up to 2.5m positions, so the comparison between them reflects the compute budget as much as the architectures themselves.

## Acknowledgements

Thanks to Professor Fergus Imrie and Silvia Sapora for supervising the project. The modelling builds on ideas from [Maia-2](https://github.com/CSSLab/maia2), AlphaZero and DeepMind's work on transformer chess engines, and all game data comes from Lichess's open database.