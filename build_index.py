"""Offline pipeline: preprocess -> index -> train classifiers -> cluster -> evaluate.

Run once:  python build_index.py
Produces artifacts/model.joblib used by the Streamlit app.
"""
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from sklearn.ensemble import IsolationForest, VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

from core import BM25, rrf, style_features

DATA_URL = ("https://huggingface.co/api/datasets/ErfanMoosaviMonazzah/"
            "fake-news-detection-dataset-English/parquet/default/train/0.parquet")
RAW = Path("data/news.parquet")
OUT = Path("artifacts/model.joblib")
N_DOCS = 6000
N_TOPICS = 10
SEED = 42


def load():
    if not RAW.exists():
        RAW.parent.mkdir(exist_ok=True)
        print("Downloading dataset ...")
        pd.read_parquet(DATA_URL).to_parquet(RAW)
    df = pd.read_parquet(RAW)[["title", "text", "subject", "date", "label"]]
    df = df.dropna().drop_duplicates("title")
    # Dataset convention: 0 = fake, 1 = real. We store misleading=1.
    df["misleading"] = 1 - df["label"]
    # Remove the "CITY (Reuters) -" byline: it leaks the label.
    df["text"] = df["text"].str.replace(r"^.{0,80}?\(Reuters\)\s*-\s*", "", regex=True)
    df = df[df["text"].str.len() > 200]
    df = df.groupby("misleading").sample(N_DOCS // 2, random_state=SEED)
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def clean(s):
    return re.sub(r"\s+", " ", re.sub(r"http\S+", " ", s)).strip()


def main():
    t0 = time.time()
    df = load()
    df["text"] = df["text"].map(clean)
    docs = (df["title"] + ". " + df["text"]).tolist()
    print(f"Loaded {len(df)} articles")

    # ---------- Sparse + dense indexes ----------
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                            min_df=3, max_df=0.6, sublinear_tf=True, max_features=60000)
    X_tfidf = tfidf.fit_transform(docs)
    bm25 = BM25(docs)
    svd = TruncatedSVD(256, random_state=SEED)
    X_dense = normalize(svd.fit_transform(X_tfidf)).astype(np.float32)
    print("Indexed (BM25 + TF-IDF + dense embeddings)")

    # ---------- Fake-news ensemble ----------
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.25, stratify=df["misleading"], random_state=SEED)
    y = df["misleading"].values
    models = {
        "Logistic Regression": LogisticRegression(C=4, max_iter=2000),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Linear SVM (calibrated)": CalibratedClassifierCV(LinearSVC(C=0.5), cv=3),
    }
    ensemble = VotingClassifier(list(models.items()), voting="soft")
    ensemble.fit(X_tfidf[tr], y[tr])

    clf_metrics = []
    for name, est in list(ensemble.named_estimators_.items()) + [("Soft-Voting Ensemble", ensemble)]:
        p = est.predict_proba(X_tfidf[te])[:, 1]
        pred = (p >= 0.5).astype(int)
        clf_metrics.append(dict(Model=name, Accuracy=accuracy_score(y[te], pred),
                                Precision=precision_score(y[te], pred), Recall=recall_score(y[te], pred),
                                F1=f1_score(y[te], pred), ROC_AUC=roc_auc_score(y[te], p)))
    cm = confusion_matrix(y[te], ensemble.predict(X_tfidf[te]))
    df["p_misleading"] = ensemble.predict_proba(X_tfidf)[:, 1]
    print("Ensemble trained:", round(clf_metrics[-1]["F1"], 4), "F1")

    # ---------- Topic modeling (NMF) + clustering ----------
    nmf = NMF(N_TOPICS, random_state=SEED, init="nndsvd", max_iter=400)
    W = nmf.fit_transform(X_tfidf)
    vocab = np.array(tfidf.get_feature_names_out())
    topic_words = [vocab[c.argsort()[::-1][:8]].tolist() for c in nmf.components_]
    topic_names = []
    for c in nmf.components_:
        words = [w for w in vocab[c.argsort()[::-1][:30]] if " " not in w]
        topic_names.append(" · ".join(words[:3]).title())
    df["topic"] = W.argmax(1)
    km = KMeans(24, n_init=4, random_state=SEED).fit(X_dense)
    df["cluster"] = km.labels_

    # ---------- Near-duplicate story groups ----------
    sims = X_dense @ X_dense.T
    np.fill_diagonal(sims, 0)
    dup_of = [np.where(row > 0.92)[0].tolist() for row in sims]
    df["n_dups"] = [len(d) for d in dup_of]

    # ---------- Anomaly detection ----------
    S = np.array([list(style_features(t, b).values()) for t, b in zip(df["title"], df["text"])])
    iso = IsolationForest(contamination=0.03, random_state=SEED).fit(np.hstack([S, X_dense[:, :20]]))
    df["anomaly"] = iso.predict(np.hstack([S, X_dense[:, :20]])) == -1

    # ---------- 2-D projection ----------
    xy = TSNE(2, init="pca", perplexity=35, random_state=SEED).fit_transform(
        PCA(50, random_state=SEED).fit_transform(X_dense))
    df["x"], df["y"] = xy[:, 0], xy[:, 1]
    print("Topics, clusters, duplicates, anomalies, t-SNE done")

    # ---------- Retrieval evaluation (known-item: headline -> article body) ----------
    rng = np.random.default_rng(SEED)
    q_ids = rng.choice(len(df), 300, replace=False)
    bodies = df["text"].tolist()
    e_tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2, max_df=0.6)
    E = e_tfidf.fit_transform(bodies)
    e_bm25 = BM25(bodies)
    e_svd = TruncatedSVD(256, random_state=SEED).fit(E)
    e_dense = normalize(e_svd.transform(E))

    def ranks(scores):
        return np.argsort(-scores)[:100]

    runs = {"BM25": [], "TF-IDF": [], "Dense (LSA)": [], "Hybrid RRF": []}
    for q in q_ids:
        query = df["title"].iat[q]
        qv = e_tfidf.transform([query])
        r_bm = ranks(e_bm25.score(query))
        r_tf = ranks((E @ qv.T).toarray().ravel())
        r_de = ranks(e_dense @ normalize(e_svd.transform(qv)).ravel())
        runs["BM25"].append(r_bm)
        runs["TF-IDF"].append(r_tf)
        runs["Dense (LSA)"].append(r_de)
        runs["Hybrid RRF"].append(np.array(rrf([r_bm, r_de, r_tf]))[:100])

    ir_metrics = []
    for name, rs in runs.items():
        pos = [np.where(r == q)[0] for r, q in zip(rs, q_ids)]
        pos = [p[0] + 1 if len(p) else None for p in pos]
        ir_metrics.append({
            "Retriever": name,
            "MRR": np.mean([1 / p if p else 0 for p in pos]),
            "Recall@1": np.mean([bool(p and p <= 1) for p in pos]),
            "Recall@5": np.mean([bool(p and p <= 5) for p in pos]),
            "Recall@10": np.mean([bool(p and p <= 10) for p in pos]),
            "nDCG@10": np.mean([1 / np.log2(p + 1) if p and p <= 10 else 0 for p in pos]),
        })
    print("Retrieval eval:", {m["Retriever"]: round(m["MRR"], 3) for m in ir_metrics})

    OUT.parent.mkdir(exist_ok=True)
    joblib.dump(dict(
        df=df.drop(columns=["label"]), tfidf=tfidf, X_tfidf=X_tfidf, bm25=bm25, svd=svd,
        X_dense=X_dense, ensemble=ensemble, topic_words=topic_words, topic_names=topic_names,
        nmf=nmf, dup_of=dup_of, clf_metrics=pd.DataFrame(clf_metrics), cm=cm,
        ir_metrics=pd.DataFrame(ir_metrics), n_test=len(te),
    ), OUT, compress=3)
    print(f"Saved {OUT} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
