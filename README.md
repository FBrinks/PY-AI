# PY-AI

Coursework and projects from a Python / AI studies track: data analysis with NumPy, Matplotlib and Pandas, plus a small machine-learning project. Notebooks are written mostly in Swedish.

## Contents

| Path | Description |
|------|-------------|
| `kunskapskontroll_2_1_NumPy.ipynb` | NumPy exercises: arrays, broadcasting, views vs. copies, vector/matrix algebra |
| `kunskapskontroll_2_2_matplotlib.ipynb` | Matplotlib exercises: object-oriented vs. pyplot API, line, scatter and bar plots, subplots |
| `kunskapskontroll_2_3_Pandas.ipynb` | Pandas exercises on a `cars_data` dataset: loading, cleaning, filtering, sorting, grouping |
| `kunskapskontroll_3.ipynb` | Exploratory data analysis of the California `housing.csv` dataset, with reflections and self-evaluation |
| `keyword_adder/` | Multilabel text-classification project (see below) |

## keyword_adder

Predicts the top 5 suggested `Keywords` for outdoor products in a PIM (Product Information Management) system, based on product text and classification.

- `eda_multilabel.ipynb` – EDA, preprocessing and training/comparison of three scikit-learn models (TF-IDF + one-vs-rest logistic regression, with and without one-hot encoded classification/SEO keyword features), evaluated with Precision@5, Recall@5 and F1.
- `keyword_adder.ipynb` – loads the saved model with `joblib` and tests predictions at different thresholds.
- `resources/dataset_keywords_swe.csv` – the product dataset.

## Setup

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter
jupyter notebook
```

Some notebooks expect their datasets (e.g. `cars_data`, `housing.csv`) in the working directory.
