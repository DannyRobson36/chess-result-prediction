"""
utils_plotting.py
Shared matplotlib style, figure sizing, and per-model colour/linestyle-assignment helpers for
evaluation plots.

Latest changes: 01/09/26:
- Tol muted palette
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

# Paul Tol's CVD-safe qualitative "muted" palette (9 colours), plus a trailing mid-grey for overflow
# beyond 9 names. The grey is not part of the CVD-safe set and is only used as a last resort.
PALETTE = (
    '#332288',  # indigo
    '#88CCEE',  # cyan
    '#44AA99',  # teal
    '#117733',  # green
    '#999933',  # olive
    '#DDCC77',  # sand
    '#CC6677',  # rose
    '#882255',  # wine
    '#AA4499',  # purple
    '#888888',  # grey, overflow only
)

# Default linestyle assigned by linestyles_for when no override is given.
DEFAULT_LINESTYLE = '-'

####################
# FUNCTIONS
####################

# (a) STYLE

def apply_plot_style(overrides: dict | None = None) -> None:
    """Applies PLOT_STYLE to matplotlib's rcParams, merged with any overrides."""
    plt.rcParams.update({**PLOT_STYLE, **(overrides or {})})


# (b) COLOUR / LINESTYLE ASSIGNMENT

def colors_for(names: list[str], canonical_order: list[str] | None = None,
                overrides: list[str | None] | None = None, palette: tuple = PALETTE) -> dict[str, str]:
    """Assigns each name in names a colour from palette. If canonical_order is given, a name's colour
    is fixed by its position in canonical_order, stable across plots and subsets; names absent from
    canonical_order fall back to a cycling position after canonical_order's length. If canonical_order
    is None, colours are assigned by position in names. overrides, given by position in names (None
    entries keep the assigned colour), force specific names to a specific colour."""
    assigned = {}
    next_fallback = len(canonical_order) if canonical_order is not None else 0
    for i, name in enumerate(names):
        if canonical_order is not None and name in canonical_order:
            idx = canonical_order.index(name)
        elif canonical_order is not None:
            idx = next_fallback
            next_fallback += 1
        else:
            idx = i
        assigned[name] = palette[idx % len(palette)]

    if overrides is not None:
        if len(overrides) != len(names):
            raise ValueError(f'overrides must have the same length as names ({len(names)}), got {len(overrides)}.')
        for name, override in zip(names, overrides):
            if override is not None:
                assigned[name] = override

    return assigned


def linestyles_for(names: list[str], overrides: list[str | None] | None = None,
                    default: str = DEFAULT_LINESTYLE) -> dict[str, str]:
    """Assigns each name in names default. overrides, given by position in names (None entries keep
    default), force specific names to a specific linestyle."""
    assigned = {name: default for name in names}
    if overrides is not None:
        if len(overrides) != len(names):
            raise ValueError(f'overrides must have the same length as names ({len(names)}), got {len(overrides)}.')
        for name, override in zip(names, overrides):
            if override is not None:
                assigned[name] = override
    return assigned