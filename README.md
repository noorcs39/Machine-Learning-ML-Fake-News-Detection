# 📰 Fake News Detection with Machine Learning

<table>
<tr>
<td>

**Binary NLP classification** — distinguish fake vs. real news articles using classical scikit-learn pipelines.

| | |
|---|---|
| **Task** | Text classification |
| **Dataset** | [Fake & Real News (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) |
| **Models** | Naive Bayes · Logistic Regression · Decision Tree · Random Forest · Linear SVM |
| **Stack** | Python · pandas · scikit-learn · NLTK · Jupyter |

</td>
<td align="right">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-NLP-FF7F0E?style=flat-square&logo=scikitlearn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-Text-059669?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-6366F1?style=flat-square)

</td>
</tr>
</table>

> Bachelor-era learning project — exploratory NLP, EDA, word clouds, confusion matrices, and side-by-side model comparison.

---

## Pipeline

```mermaid
flowchart LR
    A[Fake.csv + True.csv] --> B[Merge & shuffle]
    B --> C[Text cleaning]
    C --> D[CountVectorizer]
    D --> E[TfidfTransformer]
    E --> F{Classifier}
    F --> G[Naive Bayes]
    F --> H[Logistic Regression]
    F --> I[Decision Tree]
    F --> J[Random Forest]
    F --> K[Linear SVM]
```

Each classifier shares the same feature pipeline: bag-of-words counts → TF-IDF weighting → model fit on article **text** only (title and date removed).

---

## Project Structure

```
├── data/                  # Fake.csv & True.csv (download — see data/README.md)
├── docs/
│   ├── RESULTS.md         # Recorded model accuracies
│   └── images/            # Sample EDA & evaluation plots
├── notebooks/
│   └── fake_news_detection.ipynb
├── outputs/               # Generated results (gitignored)
├── scripts/
│   └── train.py           # CLI training & evaluation
├── src/
│   └── fake_news_detection/
│       ├── data.py
│       ├── models.py
│       └── preprocessing.py
├── tests/
├── LICENSE
├── pyproject.toml
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
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
pip install -e .
```

Download `Fake.csv` and `True.csv` from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) into the `data/` folder.

---

## Usage

**Jupyter notebook** (full EDA, word clouds, confusion matrices):

```bash
jupyter notebook notebooks/fake_news_detection.ipynb
```

**Command-line training** (requires dataset in `data/`):

```bash
python scripts/train.py --data-dir data
```

**Run tests:**

```bash
pytest tests/ -v
```

---

## Results

| Model | Accuracy |
| --- | ---: |
| Naive Bayes | 94.82% |
| Logistic Regression | 99.04% |
| Decision Tree | 99.72% |
| Random Forest | — |
| Linear SVM | — |

See [docs/RESULTS.md](docs/RESULTS.md) for details. Re-run the notebook or `scripts/train.py` after adding the dataset to reproduce metrics.

---

## Sample Outputs

<p align="center">
  <img src="docs/images/img1.png" alt="EDA sample 1" width="280" />
  <img src="docs/images/img2.png" alt="EDA sample 2" width="280" />
  <img src="docs/images/img3.png" alt="EDA sample 3" width="280" />
</p>
<p align="center">
  <img src="docs/images/img4.png" alt="Word cloud sample 1" width="280" />
  <img src="docs/images/img5.png" alt="Word cloud sample 2" width="280" />
  <img src="docs/images/img6.png" alt="Confusion matrix sample" width="280" />
</p>
<p align="center">
  <img src="docs/images/img7.png" alt="Model comparison 1" width="280" />
  <img src="docs/images/img8.png" alt="Model comparison 2" width="280" />
  <img src="docs/images/img9.png" alt="Accuracy chart" width="280" />
</p>

---

## Author

**Noor Uddin**  
📧 [noor.cs2@yahoo.com](mailto:noor.cs2@yahoo.com)  
🔗 [linkedin.com/in/noonari](https://www.linkedin.com/in/noonari/)  
🐙 [github.com/noorcs39](https://github.com/noorcs39)

---

## License

This project is licensed under the [MIT License](LICENSE).
