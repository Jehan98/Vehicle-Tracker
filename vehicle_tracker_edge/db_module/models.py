from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime
from .db import db


class SearchJob(db):
    " Search Job "
    __tablename__ = 'search_jobs'

    id = Column(Integer, primary_key=True)
    vehicle_plate = Column(String(20), nullable=False)
    vehicle_color = Column(String(20), nullable=False)
    search_duration = Column(Integer, nullable=False)
    description = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
