"""Shared IR building blocks: BM25, rank fusion, stylometric features, live RSS."""
import re
import xml.etree.ElementTree as ET

import numpy as np
import requests
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

CLICKBAIT = {"shocking", "you won't believe", "breaking", "must see", "watch", "exposed",
             "destroys", "slams", "bombshell", "secret", "truth", "busted", "epic", "wow"}


class BM25:
    """Okapi BM25 over a sparse term-frequency matrix."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.vec = CountVectorizer(stop_words="english", min_df=2)
        tf = self.vec.fit_transform(docs).tocsc().astype(np.float32)
        n, dl = tf.shape[0], np.asarray(tf.sum(1)).ravel()
        df = np.diff(tf.indptr)
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        tf = tf.tocoo()
        norm = k1 * (1 - b + b * dl[tf.row] / dl.mean())
        w = tf.data * (k1 + 1) / (tf.data + norm) * self.idf[tf.col]
        self.W = sparse.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape).tocsc()

    def score(self, query):
        q = self.vec.transform([query]).indices
        return np.asarray(self.W[:, q].sum(1)).ravel()


def rrf(rankings, k=60):
    """Reciprocal Rank Fusion of several ranked id lists."""
    scores = {}
    for r in rankings:
        for pos, doc in enumerate(r):
            scores[doc] = scores.get(doc, 0) + 1 / (k + pos + 1)
    return sorted(scores, key=scores.get, reverse=True)


def style_features(title, body):
    words = body.split() or [""]
    t_letters = [c for c in title if c.isalpha()] or ["a"]
    low = (title + " " + body[:500]).lower()
    return {
        "Headline CAPS ratio": sum(c.isupper() for c in t_letters) / len(t_letters),
        "Exclamation density": (title + body).count("!") / max(len(words), 1) * 100,
        "Question marks": title.count("?") + body.count("?"),
        "Clickbait terms": sum(k in low for k in CLICKBAIT),
        "Avg word length": float(np.mean([len(w) for w in words])),
        "Article length (words)": len(words),
        "First/second-person %": sum(w.lower() in {"i", "we", "you", "our", "your"} for w in words)
                                 / len(words) * 100,
        "Quoted sources": body.count("said") + body.count("according to"),
    }


FEEDS = {
    "BBC World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "BBC Technology": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "NPR News": "https://feeds.npr.org/1001/rss.xml",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
}


def fetch_rss(url, limit=15):
    xml = requests.get(url, timeout=10, headers={"User-Agent": "VeriNewsAI/1.0"}).content
    items = []
    for it in ET.fromstring(xml).iter("item"):
        desc = re.sub(r"<[^>]+>", " ", it.findtext("description") or "")
        img = None
        for el in it:
            if el.tag.endswith(("thumbnail", "content")) and el.get("url"):
                img = el.get("url")
        items.append(dict(title=(it.findtext("title") or "").strip(), text=desc.strip(),
                          link=it.findtext("link"), date=it.findtext("pubDate"), image=img))
        if len(items) >= limit:
            break
    return items
