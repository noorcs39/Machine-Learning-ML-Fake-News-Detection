"""Fake news detection with classical ML and scikit-learn pipelines."""

from fake_news_detection.data import load_and_merge
from fake_news_detection.models import MODEL_NAMES, build_pipeline
from fake_news_detection.preprocessing import preprocess_dataframe, punctuation_removal

__all__ = [
    "MODEL_NAMES",
    "build_pipeline",
    "load_and_merge",
    "preprocess_dataframe",
    "punctuation_removal",
]
