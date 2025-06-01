import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score

class Evaluator:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        self.detections = []
        self.ground_truths = []
        self.matches = []
        self.id_switches = 0

    def add_detections(self, frame_idx, tracks):
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            class_name = track['class_name'].replace('emergency_', '')
            conf = track['confidence']
            track_id = track['track_id']
            self.detections.append([frame_idx, x1, y1, x2, y2, class_name, conf, track_id])
        print(f"Frame {frame_idx}: Added {len(tracks)} detections")

    def load_ground_truth(self, gt_file):
        try:
            with open(gt_file, 'r') as f:
                df = pd.read_csv(f)
            required_columns = ['frame', 'x1', 'y1', 'x2', 'y2', 'class', 'track_id']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Ground truth CSV missing required columns. Found: {list(df.columns)}, Required: {required_columns}")
            print(f"Loaded {len(df)} ground truth annotations")
            for _, row in df.iterrows():
                self.ground_truths.append([
                    int(row['frame']),
                    float(row['x1']),
                    float(row['y1']),
                    float(row['x2']),
                    float(row['y2']),
                    str(row['class']),
                    int(row['track_id'])
                ])
        except Exception as e:
            print(f"Error loading ground truth: {e}")
            self.ground_truths = []

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

    def _match_detections(self, frame_idx):
        frame_dets = [d for d in self.detections if d[0] == frame_idx]
        frame_gts = [g for g in self.ground_truths if g[0] == frame_idx]
        matches = []
        if not frame_dets or not frame_gts:
            print(f"Frame {frame_idx}: No detections or ground truth")
            return matches, len(frame_dets), len(frame_gts)

        iou_matrix = np.zeros((len(frame_dets), len(frame_gts)))
        for i, det in enumerate(frame_dets):
            for j, gt in enumerate(frame_gts):
                iou_matrix[i, j] = self._calculate_iou(det[1:5], gt[1:5])

        det_indices, gt_indices = np.where(iou_matrix >= self.iou_threshold)
        for d_idx, g_idx in zip(det_indices, gt_indices):
            if frame_dets[d_idx][5] == frame_gts[g_idx][5]:
                matches.append((d_idx, g_idx, frame_dets[d_idx][7], frame_gts[g_idx][6]))

        false_positives = len(frame_dets) - len(matches)
        false_negatives = len(frame_gts) - len(matches)
        print(f"Frame {frame_idx}: {len(matches)} matches, {false_positives} false positives, {false_negatives} false negatives")
        return matches, false_positives, false_negatives

    def evaluate_detection(self):
        if not self.ground_truths:
            print("No ground truth provided, returning zero metrics")
            return {"precision": 0.0, "recall": 0.0, "mAP": 0.0}

        true_positives = 0
        false_positives = 0
        false_negatives = 0
        confidences = []
        labels = []

        for frame_idx in set(d[0] for d in self.detections):
            matches, fp, fn = self._match_detections(frame_idx)
            true_positives += len(matches)
            false_positives += fp
            false_negatives += fn

            frame_dets = [d for d in self.detections if d[0] == frame_idx]
            frame_gts = {tuple(g[1:5]): g[5] for g in self.ground_truths if g[0] == frame_idx}
            for det in frame_dets:
                det_box = tuple(det[1:5])
                det_conf = det[6]
                det_class = det[5]
                matched = False
                for gt_box, gt_class in frame_gts.items():
                    if self._calculate_iou(det_box, gt_box) >= self.iou_threshold and det_class == gt_class:
                        matched = True
                        break
                confidences.append(det_conf)
                labels.append(1 if matched else 0)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

        if confidences and labels:
            precision_curve, recall_curve, _ = precision_recall_curve(labels, confidences)
            ap = average_precision_score(labels, confidences)
        else:
            ap = 0.0

        print(f"Evaluation: TP={true_positives}, FP={false_positives}, FN={false_negatives}, Precision={precision:.3f}, Recall={recall:.3f}, mAP={ap:.3f}")
        return {
            "precision": precision,
            "recall": recall,
            "mAP": ap
        }

    def evaluate_tracking(self):
        if not self.ground_truths:
            print("No ground truth for tracking, returning zero metrics")
            return {"MOTA": 0.0, "ID_Switches": 0}

        self.id_switches = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        mismatches = 0
        prev_matches = {}

        for frame_idx in sorted(set(d[0] for d in self.detections)):
            matches, fp, fn = self._match_detections(frame_idx)
            true_positives += len(matches)
            false_positives += fp
            false_negatives += fn

            current_matches = {}
            for d_idx, g_idx, det_id, gt_id in matches:
                current_matches[det_id] = gt_id

            for det_id, gt_id in current_matches.items():
                if det_id in prev_matches and prev_matches[det_id] != gt_id:
                    self.id_switches += 1
                    mismatches += 1
            prev_matches = current_matches

        total_gt = len(self.ground_truths)
        mota = 1.0 - (false_negatives + false_positives + mismatches) / total_gt if total_gt > 0 else 0.0

        print(f"Tracking: MOTA={mota:.3f}, ID Switches={self.id_switches}")
        return {
            "MOTA": mota,
            "ID_Switches": self.id_switches
        }