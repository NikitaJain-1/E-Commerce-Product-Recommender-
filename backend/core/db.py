from sqlalchemy import create_engine, Column, Integer, String, Float, Text, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.sql import func
from .config import DB_URL

engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class Product(Base):
    __tablename__ = "products"
    product_id = Column(String, primary_key=True)
    title = Column(String)
    brand = Column(String)
    category = Column(String)
    price = Column(Float)
    desc = Column(Text)

class User(Base):
    __tablename__ = "users"
    user_id = Column(String, primary_key=True)

class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.user_id"))
    product_id = Column(String, ForeignKey("products.product_id"))
    event_type = Column(String)  # view, add_to_cart, purchase, wish
    ts = Column(DateTime, server_default=func.now())
    weight = Column(Float)

def init_db():
    Base.metadata.create_all(bind=engine)
