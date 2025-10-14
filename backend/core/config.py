import os
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DB_URL", "sqlite:///./ecomm.db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Event weights and recency half-life (in days)
EVENT_WEIGHTS = {
    "view": 1.0,
    "add_to_cart": 3.0,
    "purchase": 5.0,
    "wish": 2.0,
}
RECENCY_HALFLIFE_DAYS = 30
