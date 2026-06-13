"""Text preprocessing utilities for fake news classification."""

from __future__ import annotations

import string

import pandas as pd


def punctuation_removal(text: str) -> str:
    """Remove punctuation characters from a string."""
    return "".join(char for char in text if char not in string.punctuation)


def preprocess_dataframe(
    df: pd.DataFrame,
    stop_words: set[str] | None = None,
) -> pd.DataFrame:
    """Clean article text: drop metadata columns, lowercase, strip punctuation and stop words."""
    data = df.copy()
    for column in ("date", "title"):
        if column in data.columns:
            data.drop([column], axis=1, inplace=True)

    data["text"] = data["text"].apply(str.lower)
    data["text"] = data["text"].apply(punctuation_removal)

    if stop_words:
        data["text"] = data["text"].apply(
            lambda text: " ".join(word for word in text.split() if word not in stop_words)
        )

    return data
