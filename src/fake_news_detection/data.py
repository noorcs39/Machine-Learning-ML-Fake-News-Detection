"""Dataset loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.utils import shuffle


def load_and_merge(fake_path: str | Path, true_path: str | Path) -> pd.DataFrame:
    """Load Fake.csv and True.csv, label rows, and shuffle the combined dataset."""
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)
    fake["target"] = "fake"
    true["target"] = "true"
    data = pd.concat([fake, true], ignore_index=True)
    return shuffle(data).reset_index(drop=True)
