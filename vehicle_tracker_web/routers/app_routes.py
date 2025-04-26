from datetime import datetime, timedelta, timezone
import os

import pytz
from flask import Blueprint, render_template, request, redirect, url_for, jsonify
from sqlalchemy import desc

from db_module.models import SearchJob, VehicleRecord
from db_module.deps import SessionLocal

router = Blueprint("app_routes", __name__)

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@router.route("/")
def index():
    """ Landing page """
    db = SessionLocal()
    try:
        search_jobs = db.query(SearchJob).order_by(desc(SearchJob.created_at)).all()

        data = []
        for job in search_jobs:
            vehicle_records = db.query(VehicleRecord).filter(
                VehicleRecord.search_job_id == job.id
            ).order_by(desc(VehicleRecord.found_time)).all()

            data.append({
                "search_job": job,
                "vehicle_records": vehicle_records
            })
        return render_template("index.html", data=data)
    finally:
        db.close()

@router.route("/submit-search-job", methods=["POST"])
def submit_search_job():
    """ Submit search job """
    db = SessionLocal()
    try:
        utc_time = datetime.now(timezone.utc)
        kolkata_zone = pytz.timezone('Asia/Kolkata')
        kolkata_time = utc_time.astimezone(kolkata_zone)
        job = SearchJob(
            vehicle_plate=request.form["vehicle_plate"],
            vehicle_color=request.form["vehicle_color"],
            vehicle_type=request.form["vehicle_type"],
            created_at=kolkata_time,
            search_duration=int(request.form["search_duration"]),
            description=request.form["description"]
        )
        db.add(job)
        db.commit()
        return redirect(url_for("app_routes.index"))
    finally:
        db.close()

@router.route("/search-jobs", methods=["GET"])
def show_search_jobs():
    """ Show search jobs """
    db = SessionLocal()
    try:
        search_jobs = db.query(SearchJob).order_by(desc(SearchJob.created_at)).all()
        print(search_jobs)
        data = []
        for job in search_jobs:
            vehicle_records = db.query(VehicleRecord).filter(
                VehicleRecord.search_job_id == job.id
            ).order_by(desc(VehicleRecord.found_time)).all()
            data.append({
                "search_job": job,
                "vehicle_records": vehicle_records
            })
        return render_template("index.html", data=data)
    finally:
        db.close()

@router.route("/submit-vehicle", methods=["POST"])
def submit_vehicle_record():
    """ Submit found vehicle records """
    db = SessionLocal()
    try:
        file = request.files["image"]
        filename = f"{datetime.now().timestamp()}_{file.filename}"
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(image_path)

        record = VehicleRecord(
            search_job_id=int(request.form["search_job_id"]),
            found_vehicle_image_path=image_path,
            found_time=datetime.strptime(request.form["found_time"], "%Y-%m-%d %H:%M:%S"),
            description=request.form["description"],
            vehicle_plate=request.form["vehicle_plate"]
        )
        db.add(record)
        db.commit()

        search_jobs = db.query(SearchJob).order_by(desc(SearchJob.created_at)).all()

        data = []
        for job in search_jobs:
            vehicle_records = db.query(VehicleRecord).filter(
                VehicleRecord.search_job_id == job.id
            ).order_by(desc(VehicleRecord.found_time)).all()

            data.append({
                "search_job": job,
                "vehicle_records": vehicle_records
            })
        return redirect(url_for('app_routes.show_search_jobs', tab='tracking-vehicles'))
    finally:
        db.close()

@router.route("/clearall",  methods=["POST"])
def clear_all():
    """ Clear all records """
    db = SessionLocal()
    db.query(SearchJob).delete()
    db.query(VehicleRecord).delete()
    db.commit()
    return redirect(url_for('app_routes.index'))

@router.route("/pending-search-jobs", methods=["GET"])
def pending_search_jobs():
    """ Get pending search jobs """
    db = SessionLocal()
    try:
        search_jobs = db.query(SearchJob).all()
        data = []
        for job in search_jobs:
            if job.created_at is None:
                continue

            utc_time = datetime.now(timezone.utc)
            kolkata_zone = pytz.timezone('Asia/Kolkata')
            kolkata_time_now = utc_time.astimezone(kolkata_zone)
            created_at = kolkata_zone.localize(job.created_at)
            expire_time = created_at + timedelta(minutes=job.search_duration)

            print('expire_time', expire_time, kolkata_time_now)
            if expire_time < kolkata_time_now:
                continue

            job_data = {
                "id": job.id,
                "vehicle_plate": job.vehicle_plate,
                "vehicle_color": job.vehicle_color,
                "vehicle_type": job.vehicle_type,
                "search_duration": job.search_duration,
                "description": job.description,
                "created_at": job.created_at.isoformat(),
            }
            data.append(job_data)
            print('pending_search_jobs: ', data)
        return jsonify(data)
    finally:
        db.close()
