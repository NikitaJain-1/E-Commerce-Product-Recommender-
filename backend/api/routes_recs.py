from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
from ..reco.ranker import recommend_for_user
from ..reco.features import load_models
from ..llm.explain import generate_explanation

router = APIRouter()

class RecItem(BaseModel):
    product_id: str
    score: float
    title: str
    brand: str
    category: str
    price: float
    why: str = ""

class RecResponse(BaseModel):
    user_id: str
    items: List[RecItem]

@router.get("/{user_id}/recommendations", response_model=RecResponse)
async def get_recommendations(user_id: str, top_k: int = 10):
    try:
        vec, item_mat, index, popularity, cooc, meta = load_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model not ready. Train first. ({e})")

    cands = recommend_for_user(user_id, top_k=top_k)
    # Build explanations
    items = []
    # Simple user aggregate signals for explanation
    # For brevity, we derive signals heuristically from scores/meta.
    for pid, score, prod in cands:
        match_signals = {
            "brand_match": bool(prod.get("brand")),
            "category_focus": True,
            "similar_items": True,
            "price_fit": True,
        }
        why = await generate_explanation(user_id, prod, {"match_signals": match_signals})
        items.append(RecItem(
            product_id=pid, score=score,
            title=prod.get("title",""), brand=prod.get("brand",""),
            category=prod.get("category",""), price=float(prod.get("price",0.0)),
            why=why
        ))
    return RecResponse(user_id=user_id, items=items)

class ExplainIn(BaseModel):
    user_id: str
    product_id: str

class ExplainOut(BaseModel):
    why: str

@router.post("/explanations", response_model=ExplainOut)
async def post_explanation(body: ExplainIn):
    vec, item_mat, index, popularity, cooc, meta = load_models()
    if body.product_id not in index.index:
        raise HTTPException(status_code=404, detail="Product not found")
    prod = meta.loc[meta["product_id"] == body.product_id].iloc[0].to_dict()
    why = await generate_explanation(body.user_id, prod, {"match_signals": {"similar_items": True}})
    return ExplainOut(why=why)
