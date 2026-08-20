"""
utils_plotting.py
Shared matplotlib style, figure sizing, and per-model colour-assignment helpers for evaluation plots.

Latest changes: 20/08/26:
- Added BASE_FIGSIZE
"""

import matplotlib.pyplot as plt

####################
# CONSTANTS
####################

# Base matplotlib rcParams applied by apply_plot_style.
PLOT_STYLE = {
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9.5,
    'figure.dpi': 150,
}

# Default (width, height) in inches for a single-panel plot; multi-panel plots scale width per panel from this.
BASE_FIGSIZE = (7, 4.5)

# Colour cycle for per-model plot lines and legend entries.
PALETTE = plt.cm.tab20.colors

####################
# FUNCTIONS
####################

# (a) STYLE

def apply_plot_style(overrides: dict | None = None) -> None:
    """Applies PLOT_STYLE to matplotlib's rcParams, merged with any overrides."""
    plt.rcParams.update({**PLOT_STYLE, **(overrides or {})})


# (b) COLOUR ASSIGNMENT

def colors_for(names: list[str], palette: tuple = PALETTE) -> dict[str, tuple]:
    """Assigns each name a stable colour from palette, in order, cycling if more names than colours."""
    return {name: palette[i % len(palette)] for i, name in enumerate(names)}