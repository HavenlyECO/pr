"""Data augmentation utilities."""

from __future__ import annotations

import pandas as pd
import numpy as np


def augment_closing_line(
    df: pd.DataFrame,
    price_col: str = "closing_odds",
    n_augments: int = 3,
    min_shift: int = 5,
    max_shift: int = 20,
) -> pd.DataFrame:
    """Return ``df`` with additional rows where the closing line is shifted.

    For each row in the original DataFrame, ``n_augments`` copies are
    generated with ``price_col`` randomly adjusted by ±``min_shift`` to
    ±``max_shift``.
    """

    aug_rows = []
    for _, row in df.iterrows():
        for _ in range(n_augments):
            shift = np.random.choice([-1, 1]) * np.random.randint(
                min_shift, max_shift + 1
            )
            aug_row = row.copy()
            aug_row[price_col] = row[price_col] + shift
            aug_rows.append(aug_row)
    return pd.concat([df, pd.DataFrame(aug_rows)], ignore_index=True)
