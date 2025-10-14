import os, pickle, math
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from ..core.config import MODEL_DIR, EVENT_WEIGHTS, RECENCY_HALFLIFE_DAYS

VECTORIZER_P = os.path.join(MODEL_DIR, "tfidf.pkl")
ITEM_MATRIX_P = os.path.join(MODEL_DIR, "item_tfidf.npy")
ITEM_INDEX_P  = os.path.join(MODEL_DIR, "item_index.json")
POPULARITY_P  = os.path.join(MODEL_DIR, "popularity.json")
COOC_P        = os.path.join(MODEL_DIR, "cooc.npy")
PROD_META_P   = os.path.join(MODEL_DIR, "product_meta.parquet")

def recency_decay(days, half_life=RECENCY_HALFLIFE_DAYS):
    return 0.5 ** (days / half_life)

def load_models():
    with open(VECTORIZER_P, "rb") as f:
        vec = pickle.load(f)
    item_mat = np.load(ITEM_MATRIX_P)
    with open(ITEM_INDEX_P, "r") as f:
        index = pd.Series(pd.read_json(f, typ='series'))
    pop = pd.read_json(POPULARITY_P, typ='series')
    cooc = np.load(COOC_P)
    meta = pd.read_parquet(PROD_META_P)
    return vec, item_mat, index, pop, cooc, meta

def save_models(vec, item_mat, index_series, popularity_series, cooc, meta_df):
    with open(VECTORIZER_P, "wb") as f:
        pickle.dump(vec, f)
    np.save(ITEM_MATRIX_P, item_mat)
    index_series.to_json(ITEM_INDEX_P)
    popularity_series.to_json(POPULARITY_P)
    np.save(COOC_P, cooc)
    meta_df.to_parquet(PROD_META_P, index=False)

def build_user_profile(user_events_df, item_mat, index_series, now=None):
    if now is None:
        now = pd.Timestamp.utcnow()
    # weights: event weight * recency
    weights = []
    vectors = []
    for _, r in user_events_df.iterrows():
        if r["product_id"] not in index_series.index:
            continue
        idx = int(index_series[r["product_id"]])
        ts = pd.to_datetime(r["ts"]).tz_localize(None)
        days = max(0, (now.tz_localize(None) - ts).days)

        w = EVENT_WEIGHTS.get(r["event_type"], 1.0) * recency_decay(days)
        weights.append(w)
        vectors.append(item_mat[idx])
    if not vectors:
        return None
    weights = np.array(weights)
    mat = np.vstack(vectors)
    prof = (weights[:, None] * mat).sum(axis=0)
    norm = np.linalg.norm(prof) + 1e-8
    return prof / norm
