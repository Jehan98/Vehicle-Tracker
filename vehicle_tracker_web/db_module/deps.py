from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db_module.db import db

DATABASE_URL = "sqlite:///./uploads.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
db.metadata.create_all(bind=engine)