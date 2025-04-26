from datetime import datetime, timezone

import pytz
from sqlalchemy import Column, Integer, String, Text, ForeignKey, TIMESTAMP
from sqlalchemy.orm import relationship
from .db import db


class SearchJob(db):
    " Search Job "
    __tablename__ = 'search_jobs'

    id = Column(Integer, primary_key=True)
    vehicle_plate = Column(String(20), nullable=False)
    vehicle_color = Column(String(20), nullable=False)
    vehicle_type = Column(String(20), nullable=False)
    search_duration = Column(Integer, nullable=False)
    description = Column(Text, nullable=False)

    utc_time = datetime.now(timezone.utc)
    kolkata_zone = pytz.timezone('Asia/Kolkata')
    kolkata_time = utc_time.astimezone(kolkata_zone)
    created_at = Column(TIMESTAMP(timezone=True), default=kolkata_time)

    vehicle_records = relationship("VehicleRecord", backref="search_job")


class VehicleRecord(db):
    " Vehicle record found "
    __tablename__ = 'vehicle_records'

    id = Column(Integer, primary_key=True)
    search_job_id = Column(Integer, ForeignKey('search_jobs.id'), nullable=False)
    found_vehicle_image_path = Column(String(120), nullable=False)
    description = Column(Text, nullable=False)
    vehicle_plate = Column(String(20), nullable=False)

    utc_time = datetime.now(timezone.utc)
    kolkata_zone = pytz.timezone('Asia/Kolkata')
    kolkata_time = utc_time.astimezone(kolkata_zone)
    found_time = Column(TIMESTAMP(timezone=True), default=kolkata_time)
