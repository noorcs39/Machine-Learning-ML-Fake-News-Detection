"""Train and evaluate fake news classifiers from the command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from fake_news_detection.data import load_and_merge
from fake_news_detection.models import MODEL_NAMES, build_pipeline
from fake_news_detection.preprocessing import preprocess_dataframe

DISPLAY_NAMES = {
    "naive_bayes": "Naive Bayes",
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "svm": "Linear SVM",
}


def ensure_stopwords() -> set[str]:
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fake news classifiers.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing Fake.csv and True.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/results.json"),
        help="Path to save accuracy results (JSON)",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    fake_path = args.data_dir / "Fake.csv"
    true_path = args.data_dir / "True.csv"
    if not fake_path.exists() or not true_path.exists():
        raise SystemExit(
            f"Missing dataset. Place Fake.csv and True.csv in {args.data_dir}/ "
            "(see data/README.md)."
        )

    raw = load_and_merge(fake_path, true_path)
    cleaned = preprocess_dataframe(raw, stop_words=ensure_stopwords())

    x_train, x_test, y_train, y_test = train_test_split(
        cleaned["text"],
        cleaned["target"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=cleaned["target"],
    )

    results: dict[str, float] = {}
    for name in MODEL_NAMES:
        pipeline = build_pipeline(name)
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        accuracy = round(accuracy_score(y_test, predictions) * 100, 2)
        results[DISPLAY_NAMES[name]] = accuracy
        print(f"{DISPLAY_NAMES[name]:25s} {accuracy:6.2f}%")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
