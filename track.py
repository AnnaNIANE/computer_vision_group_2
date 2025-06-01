import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
import csv
import os
import time

class SimpleTracker:
    def __init__(self, max_age=30):
        self.tracks = {}
        self.next_id = 1
        self.max_age = max_age
        self.frame_count = 0
        self.log_file = "output/tracking_log.csv"
        os.makedirs("output", exist_ok=True)
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'track_id', 'class_name', 'x1', 'y1', 'x2', 'y2', 'confidence', 'speed', 'direction'])

    def _calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    def update(self, detections):
        self.frame_count += 1
        for tid, track in self.tracks.items():
            track['age'] += 1
            track['hit'] = False
            if 'prev_bbox' in track:
                prev_x = (track['prev_bbox'][0] + track['prev_bbox'][2]) / 2
                prev_y = (track['prev_bbox'][1] + track['prev_bbox'][3]) / 2
                curr_x = (track['bbox'][0] + track['bbox'][2]) / 2
                curr_y = (track['bbox'][1] + track['bbox'][3]) / 2
                dx = curr_x - prev_x
                dy = curr_y - prev_y
                distance = np.sqrt(dx**2 + dy**2)
                track['speed'] = distance
                track['direction'] = np.arctan2(dy, dx) if distance > 1 else 0
            track['prev_bbox'] = track['bbox']

        if detections and self.tracks:
            cost_matrix = np.zeros((len(detections), len(self.tracks)))
            track_ids = list(self.tracks.keys())
            for d, det in enumerate(detections):
                for t, tid in enumerate(track_ids):
                    cost_matrix[d, t] = 1 - self._calculate_iou(det['bbox'], self.tracks[tid]['bbox'])
            det_indices, track_indices = linear_sum_assignment(cost_matrix)
            matched = []
            for d, t in zip(det_indices, track_indices):
                if cost_matrix[d, t] < 0.7:
                    matched.append((d, track_ids[t]))
                    self.tracks[track_ids[t]].update({
                        'bbox': detections[d]['bbox'],
                        'class_name': detections[d]['class_name'],
                        'confidence': detections[d]['confidence'],
                        'age': 0,
                        'hit': True
                    })

            unmatched_dets = [i for i in range(len(detections)) if i not in det_indices]
        else:
            unmatched_dets = list(range(len(detections)))
            matched = []

        for d in unmatched_dets:
            self.tracks[self.next_id] = {
                'bbox': detections[d]['bbox'],
                'class_name': detections[d]['class_name'],
                'confidence': detections[d]['confidence'],
                'age': 0,
                'hit': True,
                'speed': 0,
                'direction': 0
            }
            self.tracks[self.next_id]['prev_bbox'] = self.tracks[self.next_id]['bbox']
            self.next_id += 1

        tracks_to_remove = [tid for tid, track in self.tracks.items() if track['age'] > self.max_age]
        for tid in tracks_to_remove:
            del self.tracks[tid]

        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for tid, track in self.tracks.items():
                if track['hit']:
                    x1, y1, x2, y2 = track['bbox']
                    writer.writerow([
                        self.frame_count, tid, track['class_name'], x1, y1, x2, y2, track['confidence'],
                        track['speed'], track['direction']
                    ])

        return [{'track_id': tid, **track} for tid, track in self.tracks.items() if track['hit']]

    def draw(self, frame, tracks):
        for track in tracks:
            x1, y1, x2, y2 = map(int, track['bbox'])
            class_name = track['class_name']
            track_id = track['track_id']
            conf = track['confidence']
            speed = track.get('speed', 0)
            direction = track.get('direction', 0)

            np.random.seed(track_id)
            color = tuple(int(c) for c in np.random.randint(50, 255, 3))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            id_label = f"ID: {track_id}"
            (id_w, id_h), _ = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            id_pos_y = max(y1 - 25, 10)
            cv2.putText(frame, id_label, (x1, id_pos_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            label = f"{class_name} {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_pos_y = max(y1 - 5, id_pos_y + id_h + 5)
            cv2.putText(frame, label, (x1, label_pos_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            speed_label = f"S: {speed:.1f} px/f, D: {direction * 180 / np.pi:.1f}°"
            (speed_w, speed_h), _ = cv2.getTextSize(speed_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            speed_pos_y = label_pos_y + speed_h + 5
            cv2.putText(frame, speed_label, (x1, speed_pos_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            future_x = center_x + (speed * 30) * np.cos(direction)
            future_y = center_y + (speed * 30) * np.sin(direction)
            cv2.circle(frame, (int(future_x), int(future_y)), 5, (255, 0, 0), -1)
            cv2.putText(frame, "Futur", (int(future_x), int(future_y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        return frame