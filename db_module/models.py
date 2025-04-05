from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, Numeric
from sqlalchemy.orm import relationship
from .db import db


class SearchJob(db):
    __tablename__ = 'search_jobs'

    id = Column(Integer, primary_key=True)
    vehicle_plate = Column(String(20), nullable=False)
    vehicle_color = Column(String(20), nullable=False)
    vehicle_type = Column(String(20), nullable=False)
    search_duration = Column(Integer, nullable=False)
    description = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    vehicle_records = relationship("VehicleRecord", backref="search_job")


class VehicleRecord(db):
    __tablename__ = 'vehicle_records'

    id = Column(Integer, primary_key=True)
    search_job_id = Column(Integer, ForeignKey('search_jobs.id'), nullable=False)
    found_vehicle_image_path = Column(String(120), nullable=False)
    found_time = Column(DateTime, nullable=False)
    found_vehicle_type = Column(String(20), nullable=False)
    found_vehicle_speed = Column(Numeric(20, 2), nullable=False)
    description = Column(Text, nullable=False)