# VeriNews — Fake News Detection & Intelligent News Retrieval

AI-powered news search engine that retrieves, groups, ranks and **credibility-scores** news articles,
with a Streamlit dashboard.

> Articles are flagged as *potentially misleading*, never declared "fake": a model score alone
> cannot establish whether a story is true.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python build_index.py      # downloads data, builds indexes + models (~20 s)
streamlit run app.py       # opens http://localhost:8501
```

## Features → Syllabus mapping

| Module (tab) | IR / ML technique |
|---|---|
| Search | Okapi BM25, TF-IDF vector space model, dense semantic embeddings (LSA), **hybrid retrieval with Reciprocal Rank Fusion**, topic-filtered (classification-based) retrieval, near-duplicate story grouping |
| Article Analysis | **Soft-voting ensemble** (Logistic Regression + Naive Bayes + calibrated Linear SVM), confidence score, **Explainable AI** term attributions, stylometric feature engineering (CAPS, clickbait, punctuation, pronouns, quoted sources) |
| Question Answering | Conversational search / **RAG-style** extractive question answering with citations |
| Topics | **NMF topic modeling**, K-Means clustering-based retrieval, **t-SNE / PCA** dimensionality reduction, **Isolation Forest** anomaly detection |
| Live Feed | Real-time / streaming IR: ingests BBC, NPR and Al Jazeera RSS feeds, scores each story live and links it to related archive stories |
| Evaluation | Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix; retrieval MRR, Recall@k, nDCG@10 on 300 known-item queries |

## Results (6,000-article sample)

| Model | F1 |
|---|---|
| Soft-voting ensemble | 0.977 |

| Retriever | MRR | Recall@10 |
|---|---|---|
| BM25 | 0.849 | 0.960 |
| Hybrid RRF | 0.811 | 0.933 |
| TF-IDF | 0.793 | 0.920 |
| Dense (LSA) | 0.642 | 0.857 |

## Data

ISOT Fake News dataset via Hugging Face (`ErfanMoosaviMonazzah/fake-news-detection-dataset-English`).
The `CITY (Reuters) -` byline is stripped from real articles because it leaks the label.

## Files

- `build_index.py`: offline pipeline (preprocess → index → train → cluster → evaluate)
- `core.py`: BM25, rank fusion, stylometric features, RSS ingestion
- `app.py`: Streamlit dashboard
