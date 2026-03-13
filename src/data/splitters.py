from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold


@dataclass
class FoldConfig:
    n_splits: int = 5
    random_state: int = 17
    shuffle: bool = True


def add_kfold_column(df: pd.DataFrame, config: FoldConfig, fold_col: str = "fold") -> pd.DataFrame:
    df = df.copy()
    df[fold_col] = -1
    splitter = KFold(
        n_splits=config.n_splits,
        random_state=config.random_state,
        shuffle=config.shuffle,
    )
    for fold, (_, val_idx) in enumerate(splitter.split(df)):
        df.loc[df.index[val_idx], fold_col] = fold
    return df


def add_group_kfold_column(
    df: pd.DataFrame,
    config: FoldConfig,
    group_col: str = "text_id",
    fold_col: str = "fold",
) -> pd.DataFrame:
    """GroupKFold: ensures all rows sharing the same group_col value stay in the same fold.
    This prevents tablet leakage (sibling sentences from the same text crossing train/val).

    If group_col is missing or all groups are unique, falls back to regular KFold.
    """
    df = df.copy()
    df[fold_col] = -1

    groups = df[group_col].values if group_col in df.columns else None

    # Fall back to regular KFold if groups would be trivial (all unique or missing)
    if groups is None or len(set(groups)) == len(df):
        return add_kfold_column(df, config, fold_col)

    # Shuffle groups deterministically before splitting
    rng = np.random.RandomState(config.random_state)
    unique_groups = list(set(groups))
    rng.shuffle(unique_groups)
    group_order = {g: i for i, g in enumerate(unique_groups)}
    reindexed_groups = np.array([group_order[g] for g in groups])

    splitter = GroupKFold(n_splits=config.n_splits)
    for fold, (_, val_idx) in enumerate(splitter.split(df, groups=reindexed_groups)):
        df.loc[df.index[val_idx], fold_col] = fold

    return df
