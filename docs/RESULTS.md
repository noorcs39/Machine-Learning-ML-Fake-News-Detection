# Model Results

Test split: 80/20 (`random_state=42`). Features: article `text` only (title and date removed).

| Model | Accuracy |
| --- | ---: |
| Naive Bayes | 94.82% |
| Logistic Regression | 99.04% |
| Decision Tree | 99.72% |
| Random Forest | — |
| Linear SVM | — |

Results recorded from the original notebook run. Re-run `notebooks/fake_news_detection.ipynb` after placing the dataset in `data/` to reproduce metrics and plots.
