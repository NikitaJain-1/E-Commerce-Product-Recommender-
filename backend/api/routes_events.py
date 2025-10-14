from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..core.db import SessionLocal, Event, User
from ..core.config import EVENT_WEIGHTS

router = APIRouter()

class EventIn(BaseModel):
    user_id: str
    product_id: str
    event_type: str  # view, add_to_cart, purchase, wish

@router.post("")
def post_event(evt: EventIn):
    if evt.event_type not in EVENT_WEIGHTS:
        raise HTTPException(status_code=400, detail="Invalid event_type")
    db = SessionLocal()
    # ensure user exists
    if not db.query(User).filter(User.user_id == evt.user_id).first():
        db.add(User(user_id=evt.user_id))
    e = Event(user_id=evt.user_id, product_id=evt.product_id,
              event_type=evt.event_type, weight=EVENT_WEIGHTS[evt.event_type])
    db.add(e)
    db.commit()
    db.close()
    return {"ok": True}
