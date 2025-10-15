# E-commerce Product Recommender (End-to-End)

A fully working **hybrid recommender** with **LLM-style explanations** and a **lightweight dashboard**.

- **Backend:** FastAPI (Python)
- **Storage:** SQLite via SQLAlchemy (for simplicity; you can swap to Postgres)
- **Retrieval/Ranking:** TF‑IDF content similarity + co-occurrence collaborative signals + popularity
- **Explanations:** Provider-agnostic LLM wrapper with a **rule-based fallback** (no API key required)
- **Dashboard:** Static HTML page that calls the API

---

## Quickstart

```bash
# 1) Create venv & install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Seed sample data and compute embeddings
python scripts/ingest.py
python scripts/train.py

# 3) Run the API
uvicorn backend.app:app --reload --port 8000

# 4) Open the dashboard (static) in your browser
# Just double-click: dashboard/index.html
# or serve it locally (e.g., python -m http.server in the dashboard folder)
```
---

## Architecture

```
ecomm-recs/
├─ backend/
│  ├─ app.py
│  ├─ api/
│  │  ├─ routes_events.py
│  │  └─ routes_recs.py
│  ├─ core/
│  │  ├─ config.py
│  │  └─ db.py
│  ├─ reco/
│  │  ├─ features.py
│  │  ├─ trainer.py
│  │  └─ ranker.py
│  └─ llm/
│     ├─ client.py
│     └─ explain.py
├─ data/
│  ├─ sample_products.csv
│  └─ sample_events.csv
├─ scripts/
│  ├─ ingest.py
│  └─ train.py
├─ dashboard/
│  └─ index.html
├─ requirements.txt
└─ README.md
```

### Data model

- **products**: `product_id, title, brand, category, price, desc`
- **users**: `user_id`
- **events**: `(user_id, product_id, event_type, ts, weight)`
- **models/** (generated): TF-IDF vectorizer + item matrix; co-occurrence matrix; popularity

### Recommender

- **User profile embedding**: weighted average of product TF-IDF vectors (weights by event type and recency)
- **Candidate scoring**:


  `score = 0.5 * content_cosine + 0.3 * cooc_score + 0.2 * pop_score`


  Tunable in `backend/reco/ranker.py`.

### Explanations

- Build an **evidence bundle** (top categories/brands, price range, similarity reasons)
- rule-based one-liner grounded in evidence (no hallucinations)

---

## API

### POST `/events`
Ingest behavior.
```json
{ "user_id": "u1", "product_id": "p3", "event_type": "view" }
```

### GET `/users/{user_id}/recommendations?top_k=10`
Get ranked products + explanations.
```json
{
  "user_id": "u1",
  "items": [
    {"product_id": "p7", "score": 0.84, "why": "You viewed running shoes; this Nike pair matches your preferred category and price range."}
  ]
}
```

### POST `/explanations`
Regenerate a "why" for a specific (user, product).
```json
{"user_id": "u1", "product_id": "p7"}
```

---

## Extending
- Swap SQLite to Postgres; add `pgvector` for dense embeddings
- Replace TF‑IDF with SentenceTransformers for richer semantics
- Replace co-occurrence with ALS/LightFM for stronger CF signals
- Add diversity constraints & business rules
- A/B test explanations (clarity, factuality)
  
## DEMO VIDEO 
https://drive.google.com/file/d/109KTxK3Hq_RW3SR3YJ0YKEISBpOl0Jj4/view?usp=sharing
