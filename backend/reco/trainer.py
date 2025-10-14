import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict
from ..core.db import SessionLocal, Product, Event
from .features import save_models
from ..core.config import EVENT_WEIGHTS

def train():
    # Load products
    db = SessionLocal()
    prows = db.query(Product).all()
    dfp = pd.DataFrame([{
        "product_id": p.product_id,
        "title": p.title or "",
        "brand": p.brand or "",
        "category": p.category or "",
        "price": p.price or 0.0,
        "desc": p.desc or ""
    } for p in prows])

    if dfp.empty:
        raise ValueError("No products found. Ingest data first.")

    # Build TF-IDF on title + desc + brand + category
    dfp["text"] = (
        dfp["title"].fillna("") + " " +
        dfp["desc"].fillna("")  + " " +
        dfp["brand"].fillna("") + " " +
        dfp["category"].fillna("")
    )
    vec = TfidfVectorizer(min_df=1, max_features=20000, ngram_range=(1,2))
    item_mat = vec.fit_transform(dfp["text"]).astype(np.float32).toarray()

    # Popularity from events (weighted)
    erows = db.query(Event).all()
    dfe = pd.DataFrame([{
        "user_id": e.user_id, "product_id": e.product_id,
        "event_type": e.event_type, "ts": e.ts, "weight": e.weight
    } for e in erows])
    if dfe.empty:
        pop = dfp.set_index("product_id").assign(pop=0.0)["pop"]
    else:
        pop = dfe.groupby("product_id")["weight"].sum()
        pop = pop.reindex(dfp["product_id"]).fillna(0.0)
        pop.index = dfp["product_id"]

    # Simple co-occurrence (items co-appearing for same user)
    # cooc[i,j] ~ sum over users of min(count_i, count_j)
    id2idx = {pid:i for i,pid in enumerate(dfp["product_id"])}
    n = len(id2idx)
    cooc = np.zeros((n, n), dtype=np.float32)
    if not dfe.empty:
        by_user = dfe.groupby("user_id")
        for _, g in by_user:
            counts = Counter(g["product_id"])
            items = list(counts.keys())
            for i, pi in enumerate(items):
                for pj in items[i:]:
                    ii, jj = id2idx[pi], id2idx[pj]
                    w = min(counts[pi], counts[pj])
                    cooc[ii, jj] += w
                    if ii != jj:
                        cooc[jj, ii] += w
        # Normalize rows
        row_norms = np.linalg.norm(cooc, axis=1, keepdims=True) + 1e-8
        cooc = cooc / row_norms

    # Save artifacts
    index_series = pd.Series(range(n), index=dfp["product_id"])
    save_models(vec, item_mat, index_series, pop, cooc, dfp[["product_id","title","brand","category","price","desc"]])
    db.close()
    return {
        "num_products": n,
        "vocab_size": len(vec.vocabulary_),
    }
