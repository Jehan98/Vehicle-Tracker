import cv2
import time
import threading
from ultralytics import YOLO
from utills.deep_sort import DeepSort
import torch
import os
import yaml
import easyocr
import requests
from datetime import datetime

from db_module.models import SearchJob, VehicleRecord
from db_module.deps import SessionLocal
from args import JOB_SUMBIT_ENDPOINT

os.makedirs("/home/jehan/Downloads/vehical_tracker/outs", exist_ok=True)

with open('configs/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

reader = easyocr.Reader(['en'])  # OCR reader for license plates
target_plate = config.get("target_plate", "Nai3NRU").upper()

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

# Set output path
out_video_path = "configs/output.mp4"

# Load video
if ".mp4" in input_video_file_name or ".avi" in input_video_file_name:
    cap = cv2.VideoCapture(input_video_file_name)
else:
    cap = cv2.VideoCapture(f'configs/{input_video_file_name}')

if not cap.isOpened():
    raise RuntimeError(f"Failed to open video: {input_video_file_name}")
else:
    print("ok")
# Define video writer
if save_output:
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (frame_width, frame_height))

lock = threading.Lock()

def is_center_inside_rectangle(point, rectangle):
    px, py = point
    x1, y1, x2, y2 = rectangle
    return x1 <= px <= x2 and y1 <= py <= y2

system_start_time = time.time()
frame_count = 0
try:
    while time.time() - system_start_time < 100:
        db = SessionLocal()
        records = db.query(SearchJob).all()
        print('records', records)
        start_time = time.time()
        ret, frame = cap.read()
        frame_count += 1 
        print(f'Processing frame: {frame_count}')
        if not ret:
            print("Video ended or no frame.")
            break

        results = model.predict(source=frame, conf=yolo_confidence_score, classes=yolo_required_class_ids, save=False, verbose=False, imgsz=yolo_input_img_size)
        
        if results[0].boxes is None:
            continue
        
        xywhs = results[0].boxes.xywh.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()

        # Filter detections inside rectangle
        filtered_xywhs = []
        filtered_scores = []
        filtered_class_ids = []

        for i in range(len(xywhs)):
            x, y, w, h = xywhs[i]
            if is_center_inside_rectangle((x, y), right_side_rect_coodinates):
                filtered_xywhs.append(xywhs[i])
                filtered_scores.append(scores[i])
                filtered_class_ids.append(class_ids[i])

        if len(filtered_xywhs) == 0:
            continue

        # Run DeepSORT only on filtered detections
        tracks = deepsort.update(torch.Tensor(filtered_xywhs), filtered_scores, filtered_class_ids, frame)
        
        if tracks is None or len(tracks) == 0:
            continue

        bboxes = tracks[:, :4]
        identities = tracks[:, -2]

        # Draw tracked boxes inside the rectangle
        frame_copy = frame.copy()
        with lock:
            for i, box in enumerate(bboxes):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                if is_center_inside_rectangle((cx, cy), right_side_rect_coodinates):
                    roi = frame_copy[y1:y2, x1:x2]
                    # Perform OCR on the ROI
                    ocr_results = reader.readtext(roi, detail=0)
                    print('ocr_results', ocr_results)
                    img_path = f"outs/Pframe_{frame_count}_{ocr_results}-{time.time()}.jpg"
                    cv2.imwrite(img_path, roi)
                    
                    for plate in ocr_results:
                        cleaned_plate = plate.replace(" ", "").upper()
                        if target_plate in cleaned_plate:
                            print(f"Match found: {cleaned_plate} (ID: {int(identities[i])})")
                            files = {
                                "image": open(img_path, "rb")
                            }

                            data = {
                                "search_job_id": 1,
                                "found_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "found_vehicle_type": "Car",
                                "found_vehicle_speed": 45.0,
                                "description": "Spotted at checkpoint"
                            }
                            response = requests.post(JOB_SUMBIT_ENDPOINT, files=files, data=data)
                            print("Status:", response.status_code)
                            print("Response:", response.text)
                            if draw:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                                cv2.putText(frame, f'MATCH: {cleaned_plate}', (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    if draw:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'ID: {int(identities[i])}', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        
            for i, box in enumerate(bboxes):
                x1, y1, x2, y2 = [int(coord) for coord in box]

        # Calculate and display FPS
        if draw:
            fps = 1 / (time.time() - start_time)
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.rectangle(frame, 
                          (right_side_rect_coodinates[0], right_side_rect_coodinates[1]), 
                          (right_side_rect_coodinates[2], right_side_rect_coodinates[3]), 
                          (255, 0, 0), 2)
        # Save video frame
        if save_output:
            out.write(frame)

        # Show output frame
        if show_output:
            cv2.imshow('Object Tracking', frame)

except KeyboardInterrupt:
    print("Keyboard interrupt. Stopping...")

finally:
    cap.release()
    if save_output:
        out.release()
        print("Video saved successfully.")
    if show_output:
        cv2.destroyAllWindows()
    print("Resources released successfully.")
