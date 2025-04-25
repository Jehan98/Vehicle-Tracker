from datetime import datetime, timezone

import pytz
from sqlalchemy import Column, Integer, String, Text, TIMESTAMP
from .db import db


class VehicleRecord(db):
    " Vehicle record found "
    __tablename__ = 'vehicle_records'

    id = Column(Integer, primary_key=True)
    search_job_id = Column(Integer, nullable=False)
    image_path = Column(String(120), nullable=False)
    description = Column(Text, nullable=False)

    utc_time = datetime.now(timezone.utc)
    kolkata_zone = pytz.timezone('Asia/Kolkata')
    kolkata_time = utc_time.astimezone(kolkata_zone)
    found_time = Column(TIMESTAMP(timezone=True), default=kolkata_time)
