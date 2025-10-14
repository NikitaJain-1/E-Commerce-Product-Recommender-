import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from .client import has_llm, call_llm

def summarize_user_signals(user_events: pd.DataFrame, product_meta: Dict[str, Any]) -> Dict[str, Any]:
    cats = user_events["event_type"].value_counts().to_dict()
    by_cat = user_events.merge(pd.DataFrame([product_meta]))  # placeholder join if needed
    return {"counts": cats}

async def generate_explanation(user_id: str, product: Dict[str, Any], evidence: Dict[str, Any]) -> str:
    prompt = f'''
System: You are a concise e-commerce assistant. Use only provided evidence.
User: Generate a one-sentence reason why this product is recommended.
Constraints: 25 words or fewer. Factual. Mention at most 2 reasons.

Evidence:
User signals: {json.dumps(evidence.get("user_signals", {}))}
Product: {json.dumps(product, ensure_ascii=False)}
Match signals: {json.dumps(evidence.get("match_signals", {}))}

Output JSON: {{"why": "<sentence>"}}
    '''.strip()

    if has_llm():
        try:
            resp = await call_llm(prompt)
            data = json.loads(resp)
            if isinstance(data, dict) and "why" in data and isinstance(data["why"], str) and data["why"]:
                return data["why"][:280]
        except Exception:
            pass

    # Rule-based fallback grounded in evidence
    brand = product.get("brand") or ""
    category = product.get("category") or "this category"
    price = product.get("price")
    reason_bits = []

    signals = evidence.get("match_signals", {})
    if signals.get("brand_match"):
        reason_bits.append(f"matches your preferred brand {brand}")
    if signals.get("category_focus"):
        reason_bits.append(f"is in the category you often browse")
    if signals.get("similar_items"):
        reason_bits.append("is similar to items you viewed")
    if signals.get("price_fit"):
        reason_bits.append("fits your usual price range")
    if not reason_bits:
        reason_bits.append("is popular among users with similar tastes")

    title = product.get("title", "This item")
    why = f"{title} {(' and '.join(reason_bits))}."
    return (why[:200]).strip()
