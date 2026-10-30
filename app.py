"""VeriNews — News Retrieval & Credibility Analysis (Streamlit UI)."""
import html
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import normalize

from core import FEEDS, fetch_rss, rrf, style_features

st.set_page_config(page_title="VeriNews | News Retrieval & Credibility Analysis", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
[data-testid="stHeader"] {display: none;}
.block-container {padding-top: 1.4rem; max-width: 1240px;}
h1, h2, h3, h4 {letter-spacing: -0.01em;}

.topbar {display: flex; justify-content: space-between; align-items: flex-end;
  border-bottom: 1px solid #e3e6eb; padding-bottom: .9rem; margin-bottom: .4rem;}
.brand {font-size: 1.35rem; font-weight: 700; color: #0f2a44;}
.brand span {font-weight: 400; color: #5b6472; font-size: .95rem; margin-left: .7rem;
  padding-left: .7rem; border-left: 1px solid #d5d9e0;}
.stats {display: flex; gap: 1.8rem;}
.stat {text-align: right;}
.stat .v {font-size: 1.05rem; font-weight: 600; color: #111827;}
.stat .l {font-size: .7rem; color: #6b7280; text-transform: uppercase; letter-spacing: .06em;}

.section-note {color: #6b7280; font-size: .82rem; margin: -.2rem 0 1rem;}
.summary {color: #6b7280; font-size: .82rem; padding: .3rem 0 .6rem; border-bottom: 1px solid #e3e6eb;}

.res {padding: 1rem 0; border-bottom: 1px solid #eef0f3;}
.res-head {display: flex; justify-content: space-between; gap: 1rem; align-items: baseline;}
.res-title {font-size: 1.02rem; font-weight: 600; color: #0f2a44; line-height: 1.35;}
.res-meta {font-size: .78rem; color: #6b7280; margin: .25rem 0 .4rem;}
.res-meta b {font-weight: 500; color: #374151;}
.res-snip {font-size: .88rem; color: #374151; line-height: 1.5;}
.rank {color: #9ca3af; font-variant-numeric: tabular-nums; margin-right: .5rem; font-weight: 500;}

.pill {white-space: nowrap; font-size: .72rem; font-weight: 600; padding: 2px 8px; border-radius: 4px;
  border: 1px solid; font-variant-numeric: tabular-nums;}
.low {color: #166534; background: #effaf3; border-color: #b7e4c7;}
.med {color: #8a4b0f; background: #fff8eb; border-color: #f3d19c;}
.high {color: #9f1c1c; background: #fdf0f0; border-color: #f1b8b8;}
.tag {font-size: .72rem; color: #374151; background: #f3f5f8; border: 1px solid #e3e6eb;
  padding: 1px 7px; border-radius: 4px; margin-right: 4px;}

.panel-label {font-size: .7rem; color: #6b7280; text-transform: uppercase; letter-spacing: .06em; margin-bottom: .2rem;}
.score {font-size: 2.6rem; font-weight: 700; color: #111827; line-height: 1;}
.score small {font-size: 1rem; color: #9ca3af; font-weight: 500;}
.meter {height: 6px; background: #eef0f3; border-radius: 3px; margin: .8rem 0 .5rem; position: relative;}
.meter > div {height: 100%; border-radius: 3px;}
.kpi .v {font-size: 1.6rem; font-weight: 700; color: #111827; font-variant-numeric: tabular-nums;}
.kpi .l {font-size: .7rem; color: #6b7280; text-transform: uppercase; letter-spacing: .06em;}

.feed {display: flex; gap: 1rem; padding: .9rem 0; border-bottom: 1px solid #eef0f3;}
.feed img {width: 132px; height: 74px; object-fit: cover; border-radius: 4px; flex-shrink: 0; background: #eef0f3;}
.answer {border-left: 3px solid #1f4e79; padding: .2rem 0 .2rem 1rem; margin: .6rem 0 1rem;}
.answer li {margin-bottom: .45rem; color: #1f2937;}
.src {font-size: .82rem; color: #4b5563; margin-bottom: .3rem;}
.disclaimer {font-size: .75rem; color: #6b7280; border-top: 1px solid #e3e6eb; margin-top: 2rem; padding-top: .8rem;}
div[data-testid="stTabs"] button p {font-size: .9rem; font-weight: 500;}
</style>""", unsafe_allow_html=True)

ART = Path("artifacts/model.joblib")
if not ART.exists():
    st.error("Index not built yet. Run `python build_index.py` first.")
    st.stop()


@st.cache_resource
def load():
    return joblib.load(ART)


M = load()
df, tfidf, X_tfidf, X_dense, svd, bm25 = M["df"], M["tfidf"], M["X_tfidf"], M["X_dense"], M["svd"], M["bm25"]
ens, topic_names = M["ensemble"], M["topic_names"]
NAVY, RED, GREY = "#1f4e79", "#b42318", "#6b7280"


@st.cache_data
def style_table():
    S = pd.DataFrame([style_features(t, b) for t, b in zip(df["title"], df["text"])])
    S["misleading"] = df["misleading"].values
    return S.groupby("misleading").mean()


def search(query, mode, k=200):
    qv = tfidf.transform([query])
    dense = X_dense @ normalize(svd.transform(qv)).ravel()
    sparse_ = (X_tfidf @ qv.T).toarray().ravel()
    lexical = bm25.score(query)
    order = {"BM25": lexical, "Dense semantic": dense, "TF-IDF": sparse_}
    if mode in order:
        return np.argsort(-order[mode])[:k].tolist(), order[mode]
    ranked = rrf([np.argsort(-s)[:k] for s in (lexical, dense, sparse_)])[:k]
    return ranked, dense


def risk(p):
    if p >= 0.7:
        return "high", "High risk", RED
    if p >= 0.4:
        return "med", "Review", "#c98a1b"
    return "low", "Low risk", "#3f7d5c"


def pill(p):
    cls, label, _ = risk(p)
    return f'<span class="pill {cls}">{label} · {p * 100:.0f}</span>'


def result(title, body, p, meta, rank=None):
    num = f'<span class="rank">{rank:02d}</span>' if rank else ""
    st.markdown(f"""<div class="res"><div class="res-head">
      <div class="res-title">{num}{html.escape(title)}</div>{pill(p)}</div>
      <div class="res-meta">{meta}</div>
      <div class="res-snip">{html.escape(body[:300])}…</div></div>""", unsafe_allow_html=True)


def chart(fig, h=340):
    fig.update_layout(template="plotly_white", height=h, margin=dict(t=30, b=10, l=10, r=10),
                      font=dict(family="Inter, sans-serif", size=12, color="#374151"),
                      legend=dict(orientation="h", y=1.08, x=0, title=""), title_font_size=13)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def explain(text, n=10):
    lr = ens.named_estimators_["Logistic Regression"]
    contrib = tfidf.transform([text]).multiply(lr.coef_[0]).tocsr()
    terms = np.array(tfidf.get_feature_names_out())
    o = np.argsort(contrib.data)
    picks = list(o[::-1][:n // 2]) + list(o[:n // 2])
    return pd.DataFrame({"term": terms[contrib.indices[picks]], "weight": contrib.data[picks]}) \
        .drop_duplicates("term").sort_values("weight")


# ---------------- Header ----------------
updated = time.strftime("%d %b %Y", time.localtime(ART.stat().st_mtime))
st.markdown(f"""<div class="topbar">
  <div class="brand">VeriNews<span>News Retrieval &amp; Credibility Analysis</span></div>
  <div class="stats">
    <div class="stat"><div class="v">{len(df):,}</div><div class="l">Indexed articles</div></div>
    <div class="stat"><div class="v">{len(topic_names)}</div><div class="l">Topics</div></div>
    <div class="stat"><div class="v">{df.cluster.nunique()}</div><div class="l">Story clusters</div></div>
    <div class="stat"><div class="v">{int(df.anomaly.sum())}</div><div class="l">Flagged anomalies</div></div>
    <div class="stat"><div class="v">{updated}</div><div class="l">Index built</div></div>
  </div></div>""", unsafe_allow_html=True)

tabs = st.tabs(["Search", "Article Analysis", "Question Answering", "Topics", "Live Feed", "Evaluation",
                "Methodology"])

# ---------------- Search ----------------
with tabs[0]:
    st.markdown('<div class="section-note">Hybrid retrieval: Okapi BM25 + TF-IDF vector space + dense semantic '
                'embeddings, fused with Reciprocal Rank Fusion. Near-duplicate stories are collapsed.</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns([6, 2, 1])
    query = c1.text_input("Query", "russia sanctions election interference", placeholder="Search the archive")
    mode = c2.selectbox("Retrieval model", ["Hybrid (RRF)", "BM25", "Dense semantic", "TF-IDF"])
    k = c3.selectbox("Results", [5, 10, 20], index=1)
    f1, f2 = st.columns([6, 3])
    topics = f1.multiselect("Topic", topic_names, placeholder="All topics")
    f2.write("")
    hide = f2.checkbox("Exclude high-risk articles")
    if query:
        t0 = time.perf_counter()
        ranked, _ = search(query, mode)
        hits, seen = [], set()
        for i in ranked:
            r = df.iloc[i]
            if i in seen or (topics and topic_names[r.topic] not in topics) or (hide and r.p_misleading >= 0.7):
                continue
            seen.update(M["dup_of"][i])
            hits.append(i)
            if len(hits) >= k:
                break
        ms = (time.perf_counter() - t0) * 1000
        st.markdown(f'<div class="summary">{len(hits)} results · {mode} · {ms:.0f} ms</div>',
                    unsafe_allow_html=True)
        for n, i in enumerate(hits, 1):
            r = df.iloc[i]
            meta = f'{r.date} &nbsp;·&nbsp; {r.subject} &nbsp;·&nbsp; <b>{topic_names[r.topic]}</b>'
            if M["dup_of"][i]:
                meta += f' &nbsp;·&nbsp; {len(M["dup_of"][i])} similar stories grouped'
            if r.anomaly:
                meta += ' &nbsp;·&nbsp; <span class="tag">Anomalous pattern</span>'
            result(r.title, r.text, r.p_misleading, meta, n)
        if not hits:
            st.info("No results match the selected filters.")

# ---------------- Article analysis ----------------
with tabs[1]:
    st.markdown('<div class="section-note">Soft-voting ensemble (Logistic Regression, Multinomial Naive Bayes, '
                'calibrated Linear SVM) over TF-IDF n-grams, with term-level attributions and stylometric '
                'signals.</div>', unsafe_allow_html=True)
    sample = df[df.p_misleading > 0.9].iloc[3]
    a1, a2 = st.columns([1.5, 1], gap="large")
    with a1:
        title = st.text_input("Headline", sample.title)
        body = st.text_area("Article text", sample.text[:1500], height=250)
        st.button("Analyze article", type="primary")
    text = f"{title}. {body}"
    vec = tfidf.transform([text])
    p = float(ens.predict_proba(vec)[0, 1])
    cls, label, color = risk(p)
    with a2:
        with st.container(border=True):
            st.markdown(f"""<div class="panel-label">Misleading-likelihood score</div>
              <div class="score">{p * 100:.0f}<small> / 100</small></div>
              <div class="meter"><div style="width:{p * 100:.0f}%; background:{color}"></div></div>
              <span class="pill {cls}">{label}</span>""", unsafe_allow_html=True)
            st.write("")
            st.dataframe(pd.DataFrame({"Model": list(ens.named_estimators_),
                                       "P(misleading)": [e.predict_proba(vec)[0, 1]
                                                         for e in ens.named_estimators_.values()]}),
                         hide_index=True, width="stretch",
                         column_config={"P(misleading)": st.column_config.ProgressColumn(
                             format="%.2f", min_value=0, max_value=1)})
    st.write("")
    e1, e2 = st.columns([1.2, 1], gap="large")
    with e1:
        st.markdown("##### Term attributions")
        ex = explain(text)
        fig = go.Figure(go.Bar(x=ex.weight, y=ex.term, orientation="h",
                               marker_color=[RED if w > 0 else NAVY for w in ex.weight]))
        fig.update_layout(xaxis_title="Contribution (→ misleading)", yaxis_title="")
        chart(fig, 330)
    with e2:
        st.markdown("##### Stylometric signals")
        ref = style_table()
        st.dataframe(pd.DataFrame({"This article": style_features(title, body), "Credible avg": ref.loc[0],
                                   "Misleading avg": ref.loc[1]}).round(2), width="stretch")
    st.markdown("##### Related archive articles")
    sims = X_dense @ normalize(svd.transform(vec)).ravel()
    for i in np.argsort(-sims)[1:4]:
        r = df.iloc[i]
        result(r.title, r.text, r.p_misleading, f"{r.date} &nbsp;·&nbsp; cosine similarity {sims[i]:.2f}")

# ---------------- Question answering ----------------
with tabs[2]:
    st.markdown('<div class="section-note">Retrieval-augmented question answering: hybrid retrieval selects '
                'candidate articles, then evidence sentences are ranked against the question and cited.</div>',
                unsafe_allow_html=True)
    q1, q2 = st.columns([6, 1])
    q = q1.text_input("Question", "What did the court decide about the travel ban?")
    q2.write("")
    q2.button("Answer", type="primary", width="stretch")
    if q:
        ranked, _ = search(q, "Hybrid (RRF)")
        cands, seen_s = [], set()
        for i in ranked[:6]:
            for s in re.split(r"(?<=[.!?])\s+", df.text.iat[i])[:40]:
                if 40 < len(s) < 400 and s not in seen_s:
                    seen_s.add(s)
                    cands.append((s, i))
        sc = (tfidf.transform([s for s, _ in cands]) @ tfidf.transform([q]).T).toarray().ravel()
        top = np.argsort(-sc)[:3]
        items = "".join(f"<li>{html.escape(cands[j][0])} <sup>[{n + 1}]</sup></li>" for n, j in enumerate(top))
        st.markdown(f'<div class="panel-label">Answer</div><div class="answer"><ol>{items}</ol></div>'
                    '<div class="panel-label">Sources</div>', unsafe_allow_html=True)
        for n, j in enumerate(top):
            i = cands[j][1]
            st.markdown(f'<div class="src">[{n + 1}] {html.escape(df.title.iat[i])} &nbsp;·&nbsp; '
                        f'{df.date.iat[i]} &nbsp; {pill(df.p_misleading.iat[i])}</div>', unsafe_allow_html=True)

# ---------------- Topics ----------------
with tabs[3]:
    st.markdown('<div class="section-note">NMF topic modeling, K-Means story clustering, t-SNE projection of the '
                'embedding space, and Isolation Forest anomaly detection.</div>', unsafe_allow_html=True)
    color = st.segmented_control("Color by", ["Topic", "Credibility", "Story cluster"], default="Topic")
    plot = df.assign(Topic=[topic_names[t] for t in df.topic],
                     Credibility=np.where(df.p_misleading >= 0.5, "Potentially misleading", "Likely credible"),
                     **{"Story cluster": df.cluster.astype(str)})
    fig = px.scatter(plot, x="x", y="y", color=color or "Topic", hover_name="title",
                     hover_data={"x": False, "y": False}, opacity=0.7,
                     color_discrete_map={"Potentially misleading": RED, "Likely credible": NAVY})
    fig.update_traces(marker_size=4)
    fig.update_layout(xaxis_visible=False, yaxis_visible=False)
    chart(fig, 520)
    st.markdown("##### Discovered topics")
    tt = pd.DataFrame({"Topic": topic_names,
                       "Top terms": [", ".join(w) for w in M["topic_words"]],
                       "Share of corpus": [(df.topic == t).mean() for t in range(len(topic_names))],
                       "Avg. risk score": [df[df.topic == t].p_misleading.mean() for t in range(len(topic_names))]})
    st.dataframe(tt, hide_index=True, width="stretch", column_config={
        "Share of corpus": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=tt["Share of corpus"].max()),
        "Avg. risk score": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1)})
    st.markdown("##### Flagged anomalies")
    st.dataframe(df[df.anomaly][["title", "subject", "date", "p_misleading"]].head(30), hide_index=True,
                 width="stretch", column_config={
                     "title": "Headline", "subject": "Section", "date": "Date",
                     "p_misleading": st.column_config.NumberColumn("Risk score", format="%.2f")})

# ---------------- Live feed ----------------
with tabs[4]:
    st.markdown('<div class="section-note">Streaming ingestion from public RSS feeds. Each item is scored on '
                'arrival and linked to the closest story in the archive index.</div>', unsafe_allow_html=True)
    l1, l2 = st.columns([6, 1])
    feeds = l1.multiselect("Sources", list(FEEDS), default=["BBC World", "NPR News"])
    l2.write("")
    if l2.button("Fetch", type="primary", width="stretch"):
        items = []
        for f in feeds:
            try:
                items += [dict(it, source=f) for it in fetch_rss(FEEDS[f], 8)]
            except Exception as e:
                st.warning(f"{f}: {e}")
        st.session_state.live = items
    live = st.session_state.get("live", [])
    if not live:
        st.caption("Select sources and press Fetch to ingest the latest headlines.")
    for it in live:
        text = f"{it['title']}. {it['text']}"
        v = tfidf.transform([text])
        p = float(ens.predict_proba(v)[0, 1])
        rel = int(np.argmax(X_dense @ normalize(svd.transform(v)).ravel()))
        img = f'<img src="{html.escape(it["image"])}">' if it["image"] else '<img>'
        st.markdown(f"""<div class="feed">{img}<div style="flex:1">
          <div class="res-head"><div class="res-title">{html.escape(it['title'])}</div>{pill(p)}</div>
          <div class="res-meta"><b>{it['source']}</b> &nbsp;·&nbsp; {it['date'] or ''}</div>
          <div class="res-snip">{html.escape(it['text'] or '')}</div>
          <div class="res-meta" style="margin-top:.35rem">Related archive story: {html.escape(df.title.iat[rel][:90])}</div>
          </div></div>""", unsafe_allow_html=True)

# ---------------- Evaluation ----------------
with tabs[5]:
    cm_, ir = M["clf_metrics"], M["ir_metrics"]
    e = cm_.iloc[-1]
    best = ir.loc[ir.MRR.idxmax()]
    cols = st.columns(5)
    for col, (lbl, val) in zip(cols, [("Accuracy", e.Accuracy), ("F1 score", e.F1), ("ROC-AUC", e.ROC_AUC),
                                      (f"MRR · {best.Retriever}", best.MRR),
                                      (f"nDCG@10 · {best.Retriever}", best["nDCG@10"])]):
        with col.container(border=True):
            st.markdown(f'<div class="kpi"><div class="l">{lbl}</div><div class="v">{val:.3f}</div></div>',
                        unsafe_allow_html=True)
    st.write("")
    fmt = {c: st.column_config.NumberColumn(format="%.3f") for c in list(cm_.columns) + list(ir.columns)}
    l, r = st.columns([1.5, 1], gap="large")
    with l:
        st.markdown(f"##### Credibility classification · held-out test set (n = {M['n_test']:,})")
        st.dataframe(cm_, hide_index=True, width="stretch", column_config=fmt)
        fig = px.bar(cm_.melt("Model", var_name="Metric", value_name="Score"), x="Metric", y="Score",
                     color="Model", barmode="group", range_y=[0.85, 1],
                     color_discrete_sequence=["#a9c4e2", "#5b8fc7", "#6b7280", NAVY])
        chart(fig, 320)
    with r:
        st.markdown("##### Confusion matrix · ensemble")
        fig = px.imshow(M["cm"], text_auto=True, x=["Credible", "Misleading"], y=["Credible", "Misleading"],
                        color_continuous_scale=["#f3f5f8", NAVY], labels=dict(x="Predicted", y="Actual"))
        fig.update_layout(coloraxis_showscale=False)
        chart(fig, 390)
    st.markdown("##### Retrieval effectiveness · 300 known-item queries (headline → article body)")
    st.dataframe(ir, hide_index=True, width="stretch", column_config=fmt)
    fig = px.bar(ir.melt("Retriever", var_name="Metric", value_name="Score"), x="Metric", y="Score",
                 color="Retriever", barmode="group", color_discrete_sequence=[NAVY, "#5b8fc7", "#a9c4e2", "#6b7280"])
    chart(fig, 320)

# ---------------- Methodology ----------------
with tabs[6]:
    st.markdown("##### System pipeline")
    st.dataframe(pd.DataFrame([
        ("Collection & preprocessing", "ISOT news corpus, live RSS ingestion", "Label-leakage removal (Reuters byline), normalisation"),
        ("Sparse retrieval", "Okapi BM25, TF-IDF vector space model", "Uni/bi-gram index, sublinear TF"),
        ("Dense retrieval", "Semantic embeddings (LSA, 256-d)", "Truncated SVD, cosine similarity"),
        ("Hybrid ranking", "Reciprocal Rank Fusion", "Fuses BM25, TF-IDF and dense rankings"),
        ("Classification", "Ensemble learning", "Soft voting: Logistic Regression, Naive Bayes, Linear SVM"),
        ("Explainability", "Term-level attributions", "Per-term contribution to the logistic model"),
        ("Feature engineering", "Stylometric features", "Capitalisation, punctuation, clickbait lexicon, pronouns, sourcing"),
        ("Topic modeling", "Non-negative Matrix Factorization", "10 topics over TF-IDF"),
        ("Clustering", "K-Means, near-duplicate detection", "24 story clusters; cosine > 0.92 grouping"),
        ("Dimensionality reduction", "PCA → t-SNE", "2-D map of the embedding space"),
        ("Anomaly detection", "Isolation Forest", "Stylometric + embedding features, 3% contamination"),
        ("Question answering", "Retrieval-augmented extractive QA", "Evidence sentence ranking with citations"),
        ("Streaming IR", "Real-time RSS ingestion", "BBC, NPR, Al Jazeera; scored and linked on arrival"),
        ("Evaluation", "Classification and IR metrics", "Accuracy, F1, ROC-AUC, MRR, Recall@k, nDCG@10"),
    ], columns=["Component", "Technique", "Implementation"]), hide_index=True, width="stretch", height=530)

st.markdown('<div class="disclaimer">Scores indicate the likelihood that an article resembles misleading content '
            'in the training data. They are decision support for review, not a determination of truth.</div>',
            unsafe_allow_html=True)
