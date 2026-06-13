# Fake News Detection with Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/scikit--learn-NLP-FF7F0E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/Dataset-Fake%20%26%20Real%20News-059669?style=for-the-badge" alt="Dataset" />
  <img src="https://img.shields.io/badge/License-MIT-6366F1?style=for-the-badge" alt="License" />
</p>

<p align="center">
  <b>Binary text classification</b> to distinguish fake vs. real news articles using classical ML pipelines.
</p>

> Bachelor-era learning project — exploratory NLP, EDA, and model comparison with scikit-learn.

---

## Overview

This project classifies news articles as **fake** or **true** using text-only features. It loads the public Fake and Real News dataset, performs cleaning and stop-word removal, then compares several models:

- Naive Bayes
- Logistic Regression
- Decision Tree
- Random Forest
- Linear SVM

Each model uses a `CountVectorizer` → `TfidfTransformer` → classifier pipeline.

---

## Project Structure

```
├── data/                  # Place Fake.csv & True.csv here (see data/README.md)
├── docs/
│   └── RESULTS.md         # Recorded model accuracies
├── notebooks/
│   └── fake_news_detection.ipynb
├── outputs/               # Saved plots and exports
├── src/
│   └── fake_news_detection/
│       ├── data.py
│       ├── models.py
│       └── preprocessing.py
├── tests/
│   └── test_fake_news_detection.py
├── LICENSE
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/noorcs39/Machine-Learning-ML-Fake-News-Detection.git
cd Machine-Learning-ML-Fake-News-Detection
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Download `Fake.csv` and `True.csv` from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) into the `data/` folder.

---

## Usage

**Jupyter notebook (full EDA + training):**

```bash
jupyter notebook notebooks/fake_news_detection.ipynb
```

**Run tests:**

```bash
pip install -e .
set PYTHONPATH=src
pytest tests/ -v
```

---

## Results

| Model | Accuracy |
| --- | ---: |
| Naive Bayes | 94.82% |
| Logistic Regression | 99.04% |
| Decision Tree | 99.72% |

See [docs/RESULTS.md](docs/RESULTS.md) for details.

---

## Author

**Noor Uddin**  
📧 [noor.cs2@yahoo.com](mailto:noor.cs2@yahoo.com)  
🐙 [github.com/noorcs39](https://github.com/noorcs39)

---

## License

This project is licensed under the [MIT License](LICENSE).
