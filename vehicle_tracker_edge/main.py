import io
import os
import json
import time
from datetime import datetime, timezone
import threading
import pytz
import yaml
import requests
from rapidfuzz import fuzz
import cv2
import easyocr
from ultralytics import YOLO
import torch

from utills.deep_sort import DeepSort
from db_module.deps import SessionLocal
from args import JOB_SUMBIT_ENDPOINT, JOBS_ENDPOINT

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

# Initialize YOLO and DeepSORT
model = YOLO('models/yolov8n.pt')
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

VIDEO_OUT = "configs/output.mp4"

if ".mp4" in input_video_file_name or ".avi" in input_video_file_name:
    cap = cv2.VideoCapture(input_video_file_name)
else:
    cap = cv2.VideoCapture(f'configs/{input_video_file_name}')

if not cap.isOpened():
    raise RuntimeError(f"Failed to open video: {input_video_file_name}")

if save_output:
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (frame_width, frame_height))

lock = threading.Lock()

def is_center_inside_rectangle(point, rectangle):
    " Check whether a point is inside a rectangle "
    px, py = point
    x1, y1, x2, y2 = rectangle
    return x1 <= px <= x2 and y1 <= py <= y2

# Set to store track IDs of vehicles that have already been successfully OCR'd and matched
processed_vehicle_ids = set()

system_start_time = time.time()
FRAME_COUNT = 0
try:
    db = SessionLocal()
    # It's better to fetch pending search jobs less frequently or handle updates
    # potentially via a separate thread or message queue depending on application
    # requirements. For this example, we fetch once at the start.
    response = requests.get(JOBS_ENDPOINT, timeout=10)
    pending_search_jobs = json.loads(response.content)
    # You might want to validate or process pending_search_jobs structure here

    while time.time() - system_start_time < 100: # Run for a limited time
        start_time = time.time()
        ret, frame = cap.read()
        FRAME_COUNT += 1
        print(f'Processing frame: {FRAME_COUNT}')
        if not ret:
            print("Video ended or no frame.")
            break

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
        tracks = deepsort.update(torch.Tensor(filtered_xywhs), filtered_scores,
                                 filtered_class_ids, frame)

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
                        print(f'Track ID: {track_id}, OCR Results: {ocr_results}')

                        # Check against pending search jobs
                        for search_job in pending_search_jobs:
                            target_plate = search_job["vehicle_plate"].replace(" ", "").upper()
                            for plate in ocr_results:
                                cleaned_plate = plate.replace(" ", "").upper()
                                score = fuzz.partial_ratio(cleaned_plate, target_plate)

                                if score > 80:
                                    processed_vehicle_ids.add(track_id)

                                    try:
                                        success, encoded_image = cv2.imencode(".jpg", roi)
                                        image_bytes = io.BytesIO(encoded_image.tobytes())
                                        print(f"Match found for Track ID {track_id}: {cleaned_plate} (Score: {score:.2f})")

                                        files = {
                                            "image": ("frame.jpg", image_bytes, "image/jpeg")
                                        }

                                        # Prepare data for the POST request
                                        utc_time = datetime.now(timezone.utc)
                                        kolkata_zone = pytz.timezone('Asia/Kolkata')
                                        kolkata_time = utc_time.astimezone(kolkata_zone)
                                        data = {
                                            "search_job_id": search_job["id"],
                                            "found_time": kolkata_time.strftime("%Y-%m-%d %H:%M:%S"),
                                            "description": f"Spotted at checkpoint with plate {cleaned_plate}"
                                        }

                                        # Send the job submission request
                                        response = requests.post(JOB_SUMBIT_ENDPOINT, files=files,
                                                                 data=data, timeout=10)
                                        print("Job Submission Status:", response.status_code)
                                        print("Job Submission Response:", response.text)
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
        if draw:
            fps = 1 / (time.time() - start_time)
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


        # Save video frame
        if save_output:
            out.write(frame)

        # Show output frame
        if show_output:
            cv2.imshow('Object Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to exit
                 break

except KeyboardInterrupt:
    print("Keyboard interrupt. Stopping...")
except Exception as e:
    print(f"An error occurred: {e}")

finally:
    cap.release()
    if save_output:
        out.release()
        print("Video saved successfully.")
    if show_output:
        cv2.destroyAllWindows()
    # Close the database session
    if 'db' in locals() and db:
        db.close()
    print("Resources released successfully.")
