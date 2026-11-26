# PY-AI

Coursework and projects from a Python / AI studies track: data analysis with NumPy, Matplotlib and Pandas, plus a small machine-learning project. Notebooks are written mostly in Swedish.

## Contents

| Path | Description |
|------|-------------|
| `kunskapskontroll_2_1_NumPy.ipynb` | NumPy exercises: arrays, broadcasting, views vs. copies, vector/matrix algebra |
| `kunskapskontroll_2_2_matplotlib.ipynb` | Matplotlib exercises: object-oriented vs. pyplot API, line, scatter and bar plots, subplots |
| `kunskapskontroll_2_3_Pandas.ipynb` | Pandas exercises on a `cars_data` dataset: loading, cleaning, filtering, sorting, grouping |
| `kunskapskontroll_3.ipynb` | Exploratory data analysis of the California `housing.csv` dataset, with reflections and self-evaluation |
| `keyword_adder/` | End project: multilabel keyword-prediction pipeline (see below) |

## keyword_adder (end project)

A multilabel text-classification model that suggests the top 5 `Keywords` for outdoor products in a PIM (Product Information Management) system. It is meant as decision support for product enrichers, with a possible later integration into a Django app.

**Data:** a `;`-separated CSV export from the PIM (`resources/dataset_keywords_swe.csv`). Features: `Headline`, `Short Text`, `Long Text`, `SEO-Keyword`, `iPIM Classification`. Target: `Keywords` (`|`-separated, several per product).

### Workflow

1. **EDA and cleaning** (`eda_multilabel.ipynb`) – drop nulls, count keywords per product, remove the placeholder "Klassiskt friluftsliv", drop keywords used fewer than 20 times, then drop rows left without keywords.
2. **Targets** – `MultiLabelBinarizer` turns each keyword list into a binary label matrix.
3. **Split** – `train_test_split` (80/20, `random_state=42`). Encoding is fitted on the training data only, via pipelines, to avoid data leakage.
4. **Models** – three scikit-learn `Pipeline`s, all ending in `OneVsRestClassifier(LogisticRegression(solver="liblinear"))`:

| Model | Features |
|-------|----------|
| `model_one` | One combined text field (all columns) -> `TfidfVectorizer` |
| `model_two` | Text (headline, short and long text, SEO keyword) -> TF-IDF, plus iPIM classification -> `OneHotEncoder` |
| `model_three` | Text (headline, short and long text) -> TF-IDF, plus iPIM classification and SEO keyword -> separate `OneHotEncoder`s |

   TF-IDF uses unigrams and bigrams (`ngram_range=(1, 2)`, `min_df=2`, `max_df=0.95`). Preprocessing is combined with `ColumnTransformer`.
5. **Evaluation** – custom Precision@k, Recall@k and F1 on the top-k ranked keywords (k = 1, 3, 5, 10, 15).
6. **Persistence** – models and label binarizers are saved with `joblib`.
7. **Inference** (`keyword_adder.ipynb`) – loads `model_three`, ranks keywords with `predict_proba`, and splits them into a "top keyword" (probability >= 0.8) and "tail keywords" (0.25-0.8).

### Results (test set)

| Model | Precision@5 | Recall@5 | F1@5 |
|-------|-------------|----------|------|
| `model_one` | 0.286 | 0.793 | 0.421 |
| `model_two` | 0.320 | 0.878 | 0.469 |
| `model_three` | **0.338** | **0.926** | **0.496** |

Splitting the inputs into separate features improved results more than changing the model type did. `model_three` is the best model, with a Recall@5 of about 0.93.

### Libraries

| Library | Used for |
|---------|----------|
| scikit-learn | `Pipeline`, `ColumnTransformer`, `TfidfVectorizer`, `OneHotEncoder`, `MultiLabelBinarizer`, `OneVsRestClassifier`, `LogisticRegression`, `train_test_split` |
| pandas | Loading, cleaning and feature building |
| NumPy | Top-k ranking and metric calculations |
| matplotlib, seaborn | EDA and Precision/Recall plots |
| joblib | Saving and loading models |

## Setup

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter
jupyter notebook
```

Some notebooks expect their datasets (e.g. `cars_data`, `housing.csv`) in the working directory.
