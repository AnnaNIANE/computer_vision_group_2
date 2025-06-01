import cv2
import numpy as np
from ultralytics import YOLO
import logging
import os
from audio import extract_audio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Detector:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5, selected_classes=None):
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.all_classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            
            "traffic light", "fire hydrant", "stop sign", "parking meter", "ambulance"
        ]
        self.class_to_id = {cls: idx for idx, cls in enumerate(self.model.names) if cls in self.all_classes}
        self.selected_classes = selected_classes if selected_classes else self.all_classes
        self.target_class_ids = list(self.class_to_id.values())  # Détecter toutes les classes disponibles
        self.siren_detected = False

    def set_video_path(self, video_path):
        if video_path and os.path.exists(video_path):
            self.siren_detected = extract_audio(video_path)

    def detect(self, frame):
        try:
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            detections = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                if conf >= self.conf_threshold and class_name in self.all_classes:
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                        'class_name': class_name
                    })
            return detections
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []

    def detect_emergency(self, frame, detections):
        emergency_detections = []
        for det in detections:
            if det['class_name'] in ['car', 'truck', 'bus', 'motorcycle', 'ambulance']:
                is_emergency = self._is_emergency_vehicle(frame, det['bbox'])
                if is_emergency or self.siren_detected:
                    det['class_name'] = f"emergency_{det['class_name']}"
            emergency_detections.append(det)
        return emergency_detections

    def _is_emergency_vehicle(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        roi = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255])) + \
                   cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
        total_pixels = roi.shape[0] * roi.shape[1]
        red_ratio = np.sum(red_mask > 0) / total_pixels
        blue_ratio = np.sum(blue_mask > 0) / total_pixels
        return red_ratio > 0.1 or blue_ratio > 0.1