from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .db import db
from .models import VehicleRecord

DATABASE_URL = "sqlite:///./track_records.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
db.metadata.create_all(bind=engine)
