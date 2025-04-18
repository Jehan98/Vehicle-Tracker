from datetime import datetime, timedelta
import os
from flask import Blueprint, render_template, request, redirect, url_for, jsonify

from db_module.models import SearchJob, VehicleRecord
from db_module.deps import SessionLocal

router = Blueprint("app_routes", __name__)

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@router.route("/", methods=["GET"])
def index():
    """ Landing page """
    db = SessionLocal()
    records = db.query(SearchJob).all()
    return render_template("index.html", data=records)

@router.route("/submit-search-job", methods=["POST"])
def submit_search_job():
    """ Submit search job """
    db = SessionLocal()
    try:
        job = SearchJob(
            vehicle_plate=request.form["vehicle_plate"],
            vehicle_color=request.form["vehicle_color"],
            vehicle_type=request.form["vehicle_type"],
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
        search_jobs = db.query(SearchJob).all()
        print(search_jobs)
        data = []
        for job in search_jobs:
            vehicle_records = db.query(VehicleRecord).filter(
                VehicleRecord.search_job_id == job.id
            ).all()
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
            found_vehicle_type=request.form["found_vehicle_type"],
            found_vehicle_speed=float(request.form["found_vehicle_speed"]),
            description=request.form["description"]
        )
        db.add(record)
        db.commit()
        return jsonify({"status": "Vehicle record stored"})
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

            expire_time = job.created_at + timedelta(minutes=job.search_duration)
            print('expire_time', expire_time, datetime.utcnow())
            if expire_time < datetime.utcnow():
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
        return jsonify(data)
    finally:
        db.close()