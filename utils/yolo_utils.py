# utils/yolo_utils.py

from ultralytics import YOLO
import torch
import numpy as np

def load_yolo_model(model_path="yolov8n.pt"):
    model = YOLO(model_path)
    if torch.cuda.is_available():
        model.to("cuda")
    return model

def analyze_image_with_yolo(model, img):
    results = model.predict(img, verbose=False, imgsz=224)

    person_count = 0
    obstacle_detected = False
    person_bboxes = []

    for r in results:
        if r.boxes.cls is None:
            continue

        for cls, bbox in zip(r.boxes.cls, r.boxes.xyxy):
            cls_id = int(cls.item())
            bbox_np = bbox.cpu().numpy()

            if np.any(np.isnan(bbox_np)):
                continue 

            if cls_id == 0:  # COCO class 0: person
                person_count += 1
                person_bboxes.append(bbox_np)
            else:
                obstacle_detected = True

    return person_count, obstacle_detected, person_bboxes
