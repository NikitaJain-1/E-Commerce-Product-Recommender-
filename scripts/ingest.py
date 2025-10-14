import pandas as pd
from backend.core.db import SessionLocal, init_db, Product, User, Event
from backend.core.config import EVENT_WEIGHTS
from pathlib import Path

def main():
    init_db()
    db = SessionLocal()
    # Load sample products
    p = Path("data/sample_products.csv")
    dfp = pd.read_csv(p)
    for _, r in dfp.iterrows():
        if not db.query(Product).filter(Product.product_id==r["product_id"]).first():
            db.add(Product(
                product_id=r["product_id"],
                title=r["title"],
                brand=r["brand"],
                category=r["category"],
                price=float(r["price"]),
                desc=r["desc"]
            ))
    # Ensure sample users + events
    epath = Path("data/sample_events.csv")
    dfe = pd.read_csv(epath, parse_dates=["ts"])
    for uid in dfe["user_id"].unique():
        if not db.query(User).filter(User.user_id==uid).first():
            db.add(User(user_id=uid))
    db.commit()
    for _, r in dfe.iterrows():
        db.add(Event(
            user_id=r["user_id"],
            product_id=r["product_id"],
            event_type=r["event_type"],
            ts=r["ts"],
            weight=EVENT_WEIGHTS.get(r["event_type"], 1.0)
        ))
    db.commit()
    db.close()
    print("Ingested products, users, events.")

if __name__ == "__main__":
    main()
