import io
import os
import json
import time
from datetime import datetime, timezone
import threading
import traceback
import pytz
import yaml
import requests
from rapidfuzz import fuzz
import cv2
import easyocr
from ultralytics import YOLO
import torch
import numpy as np

from utills.deep_sort import DeepSort
from db_module_edge.deps import SessionLocal
from args import JOB_SUMBIT_ENDPOINT, JOBS_ENDPOINT
from db_module_edge.models import VehicleRecord
from utils import is_center_inside_rectangle, is_internet_available

os.makedirs("/home/jehan/Downloads/vehical_tracker/outs", exist_ok=True)

with open('configs/config.yaml', 'r', encoding="utf-8") as file:
    config = yaml.safe_load(file)

reader = easyocr.Reader(['en'])  # OCR reader for license plates

# Main configurations
input_video_file_name = config.get("input_video_file_name", "configs/cctv.mp4")
right_side_rect_coodinates = config.get("right_side_rect_coodinates", [635, 375, 1250, 500])
draw = config.get("draw", False)
show_output = config.get("show_output", False)
save_output = config.get("save_output", False)

# YOLO model configurations
yolo_confidence_score = config.get("yolo_confidence_score", 0.3)
yolo_required_class_ids = config.get("yolo_required_class_ids", [1, 2, 3, 5, 7])
yolo_input_img_size = config.get("yolo_input_img_size", 640)

# DeepSORT configurations
max_dist = config.get("max_cosine_dist", 0.2)
nms_max_overlap = config.get("nms_max_overlap", 1.0)
max_iou_distance = config.get("max_iou_distance", 0.7)
max_age = config.get("max_age", 70)
n_init = config.get("n_init", 3)
nn_budget = config.get("nn_budget", 100)
use_cuda_for_deepsort = config.get("use_cuda_for_deepsort", False)
frames_to_read = config.get("frames_to_read", 30)

# Initialize YOLO and DeepSORT
model = YOLO('models/yolov8n.onnx')
deepsort = DeepSort(
    model_path="utills/deep_sort/deep/checkpoint/ckpt.t7",
    max_dist=max_dist,
    min_confidence=yolo_confidence_score,
    nms_max_overlap=nms_max_overlap,
    max_iou_distance=max_iou_distance,
    max_age=max_age,
    n_init=n_init,
    nn_budget=nn_budget,
    use_cuda=use_cuda_for_deepsort
)

VIDEO_OUT = "configs/output_low_fps.mp4"

if ".mp4" in input_video_file_name or ".avi" in input_video_file_name:
    cap = cv2.VideoCapture(input_video_file_name)
else:
    cap = cv2.VideoCapture(f'configs/{input_video_file_name}')

if not cap.isOpened():
    raise RuntimeError(f"Failed to open video: {input_video_file_name}")

fps_video = int(cap.get(cv2.CAP_PROP_FPS))
skip_frames = int(fps_video / frames_to_read)

if save_output:
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(VIDEO_OUT, fourcc, frames_to_read, (frame_width, frame_height))

lock = threading.Lock()
processed_vehicle_ids = set()
system_start_time = time.time()
FRAME_COUNT = 0
fps_list = []
try:
    db = SessionLocal()
    response = requests.get(JOBS_ENDPOINT, timeout=10)
    pending_search_jobs = json.loads(response.content) if response.content else []
    print('pending_search_jobs ', pending_search_jobs)
    last_job_fetch_time = time.time()

    while cap.isOpened():
        start_time = time.time()

        if start_time - system_start_time > 100:
            break

        ret, frame = cap.read()
        FRAME_COUNT += 1

        if FRAME_COUNT % skip_frames != 0:
            print(f'Skipping frame: {FRAME_COUNT}')
            continue
        print(f'Processing frame: {FRAME_COUNT}')
        if not ret:
            print("Video ended or no frame.")
            break

        found_vehicle_records = db.query(VehicleRecord).all()

        if is_internet_available() and found_vehicle_records:
            print('found_vehicle_records', found_vehicle_records)
            for found_vehicle_record in found_vehicle_records:
                search_job_id = found_vehicle_record.search_job_id
                record_id = found_vehicle_record.id
                vehicle_desc = found_vehicle_record.description
                vehicle_plate = found_vehicle_record.vehicle_plate
                found_vehicle_image_path = found_vehicle_record.found_vehicle_image_path
                found_time = found_vehicle_record.found_time.strftime("%Y-%m-%d %H:%M:%S")

                data = {
                    "search_job_id": search_job_id,
                    "found_time": found_time,
                    "description": vehicle_desc,
                    "vehicle_plate": vehicle_plate
                }
                image = cv2.imread(found_vehicle_image_path)
                success, encoded_image_from_file = cv2.imencode(".jpg", image)
                if not success:
                    continue
                image_bytes_from_file = io.BytesIO(encoded_image_from_file.tobytes())
                files = {
                    "image": (f"{search_job_id}_{vehicle_plate}.jpg", image_bytes_from_file, "image/jpeg")
                }

                response = requests.post(JOB_SUMBIT_ENDPOINT, files=files,
                                            data=data, timeout=10)
                if response.status_code == 200:
                    deleted_count = db.query(VehicleRecord).filter(
                        VehicleRecord.id == record_id).delete()
                    db.commit()
        if start_time - last_job_fetch_time >= 30:
            print("Fetching pending search jobs...")
            try:
                response = requests.get(JOBS_ENDPOINT, timeout=10)
                response.raise_for_status()
                pending_search_jobs = json.loads(response.content) if response.content else []
                last_job_fetch_time = start_time
                print('pending_search_jobs ', pending_search_jobs)
            except requests.exceptions.RequestException as e:
                print(f"Error fetching pending search jobs: {e}")

        if not pending_search_jobs:
            print("No pending search jobs")
            time.sleep(1)
            continue

        results = model.predict(source=frame, conf=yolo_confidence_score,
                                classes=yolo_required_class_ids, save=False,
                                verbose=False, imgsz=yolo_input_img_size)

        if results[0].boxes is None:
            continue

        xywhs = results[0].boxes.xywh.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()

        filtered_xywhs = []
        filtered_scores = []
        filtered_class_ids = []

        # Filter detections based on the rectangle region
        for i, xywh in enumerate(xywhs):
            x, y, w, h = xywh
            if is_center_inside_rectangle((x, y), right_side_rect_coodinates):
                filtered_xywhs.append(xywh)
                filtered_scores.append(scores[i])
                filtered_class_ids.append(class_ids[i])

        if len(filtered_xywhs) == 0:
            # If no vehicles in the desired region, continue to next frame
            continue

        # Run DeepSORT only on filtered detections within the region
        s_t = time.time()
        filtered_xywhs_array = np.array(filtered_xywhs)
        filtered_xywhs_tensor = torch.from_numpy(filtered_xywhs_array)
        tracks = deepsort.update(filtered_xywhs_tensor, filtered_scores, filtered_class_ids,
                                 frame)
        print('time for tracks: ', time.time()-s_t)
        if tracks is None or len(tracks) == 0:
            # If no tracks are updated/confirmed in the region, continue
            continue

        # tracks format: [x1, y1, x2, y2, track_id, class_id, confidence_score]
        bboxes = tracks[:, :4]
        identities = tracks[:, 4]

        # Process tracked vehicles within the rectangle for OCR
        frame_copy = frame.copy()
        with lock:
            for i, box in enumerate(bboxes):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                track_id = int(identities[i])

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Check if the vehicle's center is within the rectangle
                # AND if we haven't already processed this track ID
                if is_center_inside_rectangle((cx, cy), right_side_rect_coodinates) and track_id not in processed_vehicle_ids:
                    roi = frame_copy[y1:y2, x1:x2]
                    if roi.shape[0] > 0 and roi.shape[1] > 0: # Ensure ROI is valid
                        ocr_results = reader.readtext(roi, detail=0)
                        if not ocr_results:
                            continue
                        print(f'Track ID: {track_id}, OCR Results: {ocr_results}')

                        # Check against pending search jobs
                        for search_job in pending_search_jobs:
                            target_plate = search_job["vehicle_plate"].replace(" ", "").upper()
                            for plate in ocr_results:
                                if len(plate) < 4:
                                    print(f"detected plate is too short: {plate}")
                                    continue
                                cleaned_plate = plate.replace(" ", "").upper()
                                score = fuzz.partial_ratio(cleaned_plate, target_plate)
                                print(f"Plate match score: {plate}")
                                if score > 80:
                                    processed_vehicle_ids.add(track_id)

                                    try:
                                        success, encoded_image = cv2.imencode(".jpg", roi)
                                        image_bytes = io.BytesIO(encoded_image.tobytes())
                                        print(f"Match found for Track ID {track_id}: {cleaned_plate} (Score: {score:.2f})")

                                        files = {
                                            "image": ("frame.jpg", image_bytes, "image/jpeg")
                                        }

                                        vehicle_desc = f"Spotted at checkpoint with plate {cleaned_plate}"

                                        utc_time = datetime.now(timezone.utc)
                                        kolkata_zone = pytz.timezone('Asia/Kolkata')
                                        kolkata_time = utc_time.astimezone(kolkata_zone)
                                        found_time = kolkata_time.strftime("%Y-%m-%d %H:%M:%S")
                                        search_job_id = search_job["id"]
                                        data = {
                                            "search_job_id": search_job_id,
                                            "found_time": found_time,
                                            "description": vehicle_desc,
                                            "vehicle_plate": cleaned_plate
                                        }
                                        try:
                                            response = requests.post(JOB_SUMBIT_ENDPOINT, files=files,
                                                                 data=data, timeout=10)
                                            if response.status_code != 200:
                                                raise Exception("Error in submitting found vehicle. Storing locally...")
                                        except Exception as e:
                                            print(f"Error submitting job for Track ID {track_id}: {e}")
                                            save_directory = "./outs"
                                            image_filename = f"{search_job_id}_{cleaned_plate}.jpg"
                                            local_image_path = os.path.join(save_directory, image_filename)

                                            os.makedirs(save_directory, exist_ok=True)

                                            cv2.imwrite(local_image_path, roi)
                                            print(f"Image saved to: {local_image_path}")
                                            job = VehicleRecord(
                                                vehicle_plate=cleaned_plate,
                                                search_job_id=search_job_id,
                                                found_vehicle_image_path=local_image_path,
                                                description=vehicle_desc,
                                                found_time=kolkata_time
                                            )
                                            db.add(job)
                                            db.commit()
                                            print("Job Store Status:", response.status_code)
                                    except Exception as e:
                                        print(f"Error submitting job for Track ID {track_id}: {e}")
                    if draw:
                         # Draw box with track ID for all tracked vehicles in the region
                        color = (0, 255, 0) # Green for tracked
                        # Change color to yellow if plate was successfully processed
                        if track_id in processed_vehicle_ids:
                             color = (0, 255, 255) # Yellow for processed/matched

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        # Add matched plate text if processed
                        if track_id in processed_vehicle_ids and ocr_results:
                             display_plate = next((p for p in ocr_results if fuzz.partial_ratio(p.replace(" ", "").upper(), target_plate) > 80), "Matched")
                             cv2.putText(frame, f'{display_plate}', (x1, y2 + 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw the region of interest rectangle
        if draw:
            cv2.rectangle(frame,
                          (right_side_rect_coodinates[0], right_side_rect_coodinates[1]),
                          (right_side_rect_coodinates[2], right_side_rect_coodinates[3]),
                          (255, 0, 0), 2) # Blue rectangle

        # Calculate and display FPS
        fps = (1 + skip_frames) / (time.time() - start_time)
        print('fps: ', fps)
        fps_list.append(fps)
        if draw:
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        if save_output:
            out.write(frame)
    mean_fps = np.mean(fps_list)
    print('mean_fps: ', mean_fps)
except Exception as e:
    traceback.print_exc()
finally:
    cap.release()
    if save_output:
        out.release()
        print("Video saved successfully.")
    if show_output:
        cv2.destroyAllWindows()
    if 'db' in locals() and db:
        db.close()
    print("Resources released successfully.")
