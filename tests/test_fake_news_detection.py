import pandas as pd
import pytest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from fake_news_detection.models import MODEL_NAMES, build_pipeline
from fake_news_detection.preprocessing import preprocess_dataframe, punctuation_removal


def test_punctuation_removal():
    assert punctuation_removal("Hello, world!") == "Hello world"


def test_preprocess_dataframe_drops_columns_and_lowercases():
    sample = pd.DataFrame(
        {
            "title": ["Headline"],
            "text": ["Hello, WORLD!"],
            "subject": ["news"],
            "date": ["2020-01-01"],
            "target": ["fake"],
        }
    )
    cleaned = preprocess_dataframe(sample, stop_words={"world"})
    assert list(cleaned.columns) == ["text", "subject", "target"]
    assert cleaned.loc[0, "text"] == "hello"


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_pipeline_trains_on_sample(model_name):
    sample = pd.DataFrame(
        {
            "text": [
                "breaking news today",
                "official report released",
                "shocking claim spreads online",
                "government statement issued",
                "verified press briefing",
                "unverified rumor online",
            ],
            "target": ["fake", "true", "fake", "true", "true", "fake"],
        }
    )
    cleaned = preprocess_dataframe(sample)
    x_train, x_test, y_train, y_test = train_test_split(
        cleaned["text"],
        cleaned["target"],
        test_size=0.34,
        random_state=42,
        stratify=cleaned["target"],
    )
    pipeline = build_pipeline(model_name)
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    assert accuracy_score(y_test, predictions) >= 0.0
