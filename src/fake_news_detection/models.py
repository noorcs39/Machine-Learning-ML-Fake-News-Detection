"""Scikit-learn model pipelines used in the notebook."""

from __future__ import annotations

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

MODEL_NAMES = (
    "naive_bayes",
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "svm",
)


def build_pipeline(model_name: str) -> Pipeline:
    """Return a CountVectorizer + TfidfTransformer pipeline for the requested model."""
    vectorizer = CountVectorizer()
    tfidf = TfidfTransformer()

    if model_name == "naive_bayes":
        estimator = MultinomialNB()
    elif model_name == "logistic_regression":
        estimator = LogisticRegression(max_iter=1000)
    elif model_name == "decision_tree":
        estimator = DecisionTreeClassifier(criterion="entropy", max_depth=100)
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(n_estimators=50, criterion="entropy")
    elif model_name == "svm":
        estimator = svm.SVC(kernel="linear")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline(
        [
            ("vect", vectorizer),
            ("tfidf", tfidf),
            ("model", estimator),
        ]
    )
