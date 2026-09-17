"""
utils_latex.py
Converts pandas DataFrames into booktabs-style LaTeX table/subtable blocks for direct input into
dissertation.

Latest changes: 17/09/26:
- Shortened docstrings
"""

import math
import re

####################
# CONSTANTS
####################

# Placeholder shown for NaN or missing values.
NAN_PLACEHOLDER = '--'

# Fallback format keys for a column_spec.
DEFAULT_COLUMN_FORMAT = {
    'decimals': 2,
    'multiply': None,
    'suffix': '',
    'sign': False,
    'text': False,
}

# Fraction of \textwidth split across subtables in a grid row.
GRID_WIDTH_BUDGET = 0.96

# Characters escaped by _escape_latex_text.
LATEX_SPECIAL_CHARS = '%&#_'

####################
# FUNCTIONS
####################

# (a) TEXT ESCAPING

def _escape_latex_text(text: str) -> str:
    """Escapes unescaped LATEX_SPECIAL_CHARS characters in text."""
    return re.sub(f'(?<!\\\\)([{re.escape(LATEX_SPECIAL_CHARS)}])', r'\\\1', text)


# (b) VALUE FORMATTING

def format_value(value: float, decimals: int = 2, multiply: float | None = None, suffix: str = '',
                  sign: bool = False, nan_placeholder: str = NAN_PLACEHOLDER) -> str:
    """Formats value to decimals d.p., with optional multiply, suffix, and forced sign; returns
    nan_placeholder for NaN."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return _escape_latex_text(nan_placeholder)
    scaled = value * multiply if multiply is not None else value
    text = f'{scaled:+.{decimals}f}' if sign else f'{scaled:.{decimals}f}'
    return f'{text}{_escape_latex_text(suffix)}'


def format_text_value(value, nan_placeholder: str = NAN_PLACEHOLDER) -> str:
    """Passes value through as an escaped string; returns nan_placeholder for None/NaN."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return _escape_latex_text(nan_placeholder)
    return _escape_latex_text(str(value))


def _indent(text: str, spaces: int) -> str:
    """Indents every non-empty line in text by spaces."""
    pad = ' ' * spaces
    return '\n'.join(f'{pad}{line}' if line else line for line in text.split('\n'))


def _format_row_cells(row_label, row, column_specs: list[dict], nan_placeholder: str) -> list[str]:
    """Formats row_label and each column_spec's value in row into cell strings.

    column_specs: text=True formats that column as plain text instead of numeric.
    """
    cells = [_escape_latex_text(str(row_label))]
    for spec in column_specs:
        fmt = {**DEFAULT_COLUMN_FORMAT, **spec}
        if fmt['text']:
            cells.append(format_text_value(
                row[spec['key']],
                nan_placeholder=spec.get('nan_placeholder', nan_placeholder),
            ))
        else:
            cells.append(format_value(
                row[spec['key']],
                decimals=fmt['decimals'],
                multiply=fmt['multiply'],
                suffix=fmt['suffix'],
                sign=fmt['sign'],
                nan_placeholder=spec.get('nan_placeholder', nan_placeholder),
            ))
    return cells


# (c) TABULAR RENDERING

def render_tabular(df, column_specs: list[dict], index_label: str = 'Model',
                    nan_placeholder: str = NAN_PLACEHOLDER) -> str:
    """Renders df as a booktabs tabular block, one row per table row.

    column_specs: dicts with keys key, label, decimals, multiply, suffix, sign, text, nan_placeholder.
        text=True formats that column as plain text instead of numeric.
    """
    align = 'l' + 'r' * len(column_specs)
    header = ' & '.join([_escape_latex_text(index_label)] +
                         [_escape_latex_text(spec['label']) for spec in column_specs])

    lines = [
        f'\\begin{{tabular}}{{{align}}}',
        _indent('\\toprule', 4),
        _indent(f'{header} \\\\', 4),
        _indent('\\midrule', 4),
    ]
    for row_label, row in df.iterrows():
        cells = _format_row_cells(row_label, row, column_specs, nan_placeholder)
        lines.append(_indent(f'{" & ".join(cells)} \\\\', 4))
    lines += [_indent('\\bottomrule', 4), '\\end{tabular}']
    return '\n'.join(lines)


def render_grouped_tabular(df, column_specs: list[dict], group_specs: list[dict], index_label: str = 'Model',
                            nan_placeholder: str = NAN_PLACEHOLDER) -> str:
    """Renders df as a booktabs tabular block with a spanning group header row above the column sub-header row.

    column_specs: same as render_tabular.
    group_specs: ordered dicts with keys label, span; spans must sum to len(column_specs).
    """
    total_span = sum(g['span'] for g in group_specs)
    if total_span != len(column_specs):
        raise ValueError(f'group_specs spans sum to {total_span}, expected {len(column_specs)}.')

    align = 'l' + 'r' * len(column_specs)

    group_cells = ['']
    cmidrules = []
    col_cursor = 2
    for g in group_specs:
        group_cells.append(f"\\multicolumn{{{g['span']}}}{{c}}{{{_escape_latex_text(g['label'])}}}")
        col_end = col_cursor + g['span'] - 1
        cmidrules.append(f'\\cmidrule(lr){{{col_cursor}-{col_end}}}')
        col_cursor = col_end + 1
    group_header = ' & '.join(group_cells)
    cmidrule_line = ' '.join(cmidrules)

    sub_header = ' & '.join([_escape_latex_text(index_label)] +
                             [_escape_latex_text(spec['label']) for spec in column_specs])

    lines = [
        f'\\begin{{tabular}}{{{align}}}',
        _indent('\\toprule', 4),
        _indent(f'{group_header} \\\\', 4),
        _indent(cmidrule_line, 4),
        _indent(f'{sub_header} \\\\', 4),
        _indent('\\midrule', 4),
    ]
    for row_label, row in df.iterrows():
        cells = _format_row_cells(row_label, row, column_specs, nan_placeholder)
        lines.append(_indent(f'{" & ".join(cells)} \\\\', 4))
    lines += [_indent('\\bottomrule', 4), '\\end{tabular}']
    return '\n'.join(lines)


# (d) SUBTABLE / TABLE ASSEMBLY

def render_subtable(tabular_latex: str, caption: str, label: str, width: str = '\\textwidth') -> str:
    """Wraps tabular_latex in a top-aligned subtable of width, with its own caption and label."""
    body = '\n'.join([
        '\\centering',
        tabular_latex,
        f'\\caption{{{_escape_latex_text(caption)}}}',
        f'\\label{{{label}}}',
    ])
    return f'\\begin{{subtable}}[t]{{{width}}}\n{_indent(body, 4)}\n\\end{{subtable}}'


def render_table(subtables: list[dict], outer_caption: str, outer_label: str, n_cols: int = 1,
                  position: str = 'htbp', vspace: str = '1em') -> str:
    """Assembles subtables into a table environment.

    subtables: dicts with keys tabular, caption, label.
    n_cols: subtables per row; wraps to further rows as needed.
    """
    width = '\\textwidth' if n_cols == 1 else f'{(GRID_WIDTH_BUDGET / n_cols):.2f}\\textwidth'

    blocks = [
        render_subtable(st['tabular'], st['caption'], st['label'], width=width)
        for st in subtables
    ]

    rows = [blocks[i:i + n_cols] for i in range(0, len(blocks), n_cols)]
    row_strs = ['\\hfill\n'.join(row) for row in rows]
    body = f'\n\\vspace{{{vspace}}}\n'.join(row_strs)

    full_body = '\n'.join([
        '\\centering',
        body,
        f'\\caption{{{_escape_latex_text(outer_caption)}}}',
        f'\\label{{{outer_label}}}',
    ])
    return f'\\begin{{table}}[{position}]\n{_indent(full_body, 4)}\n\\end{{table}}'


def render_flat_table(tabular_latex: str, caption: str, label: str, position: str = 'htbp') -> str:
    """Wraps tabular_latex directly in a table environment, with its own caption and label."""
    body = '\n'.join([
        '\\centering',
        f'\\caption{{{_escape_latex_text(caption)}}}',
        f'\\label{{{label}}}',
        tabular_latex,
    ])
    return f'\\begin{{table}}[{position}]\n{_indent(body, 4)}\n\\end{{table}}'


# (e) FILE OUTPUT

def write_table(latex: str, path: str) -> None:
    """Writes a rendered table/subtable block to path as a standalone .tex file."""
    with open(path, 'w') as f:
        f.write(latex + '\n')