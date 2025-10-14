import numpy as np
import pandas as pd
from ..core.db import SessionLocal, Event
from .features import load_models, build_user_profile

ALPHA = 0.5  # content
BETA  = 0.3  # cooc
GAMMA = 0.2  # popularity

def recommend_for_user(user_id: str, top_k: int = 10):
    vec, item_mat, index, popularity, cooc, meta = load_models()

    db = SessionLocal()
    erows = db.query(Event).filter(Event.user_id == user_id).all()
    dfe = pd.DataFrame([{
        "product_id": e.product_id, "event_type": e.event_type, "ts": e.ts, "weight": e.weight
    } for e in erows])
    db.close()

    # Build user profile from events
    prof = build_user_profile(dfe, item_mat, index) if not dfe.empty else None

    # Content score
    if prof is None:
        content = np.zeros(item_mat.shape[0], dtype=np.float32)
    else:
        norms = np.linalg.norm(item_mat, axis=1) + 1e-8
        content = (item_mat @ prof) / norms

    # Cooccurrence score: average cooc to items the user interacted with
    if dfe.empty:
        cooc_score = np.zeros(item_mat.shape[0], dtype=np.float32)
    else:
        idxs = [int(index[pid]) for pid in dfe["product_id"] if pid in index.index]
        if idxs:
            cooc_score = np.mean(cooc[idxs, :], axis=0)
        else:
            cooc_score = np.zeros(item_mat.shape[0], dtype=np.float32)

    # Popularity score
    pop_aligned = popularity.reindex(index.index).fillna(0.0).values.astype(np.float32)
    if pop_aligned.max() > 0:
        pop_score = pop_aligned / pop_aligned.max()
    else:
        pop_score = pop_aligned

    # Combine
    final = ALPHA * content + BETA * cooc_score + GAMMA * pop_score

    # Remove already heavily interacted items
    seen = set(dfe["product_id"].tolist())
    candidates = []
    for pid, score in zip(index.index, final):
        if pid not in seen:
            meta_row = meta.loc[meta["product_id"] == pid].iloc[0].to_dict()
            candidates.append((pid, float(score), meta_row))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_k]
