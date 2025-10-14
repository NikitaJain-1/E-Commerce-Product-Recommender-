from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.db import init_db
from .api.routes_events import router as events_router
from .api.routes_recs import router as recs_router

app = FastAPI(title="E-comm Recs API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

app.include_router(events_router, prefix="/events", tags=["events"])
app.include_router(recs_router,   prefix="/users",  tags=["recs"])

@app.get("/health")
def health():
    return {"status": "ok"}
