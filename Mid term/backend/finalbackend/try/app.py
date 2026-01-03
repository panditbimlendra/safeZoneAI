from flask import Flask, request, send_file, jsonify
import os
import cv2
import numpy as np
from ultralytics import YOLO
import time

app = Flask(__name__)
UPLOAD_FOLDER = r"C:\Users\Bimlendra\OneDrive\Desktop\Major Project\Mid term\backend\finalbackend\carcrash"
RESULT_FOLDER = r"C:\Users\Bimlendra\OneDrive\Desktop\Major Project\Mid term\backend\finalbackend\carcrash"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

class CrashDetector:
    def __init__(
        self,
        model_path="yolov11m.pt",
        crash_threshold=0.1,
        min_distance=50,
        alert_count=2,
        conf_threshold=0.5,
    ):
        self.model = YOLO(model_path)
        self.crash_threshold = crash_threshold
        self.min_distance = min_distance
        self.alert_count = alert_count
        self.conf_threshold = conf_threshold
        self.crash_detected = False
        self.crash_frame = None

    @staticmethod
    def _calculate_iou_and_distance(box1: np.ndarray, box2: np.ndarray) -> tuple[float, float]:
        x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - inter_area
        iou = inter_area / union_area if union_area > 0 else 0

        c1 = np.array([(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2])
        c2 = np.array([(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2])
        dist = np.linalg.norm(c1 - c2)

        return iou, dist

    def detect_crash_in_frame(self, boxes: np.ndarray, classes: np.ndarray, confidences: np.ndarray) -> bool:
        # Classes: 2=car, 3=motorcycle, 5=bus, 7=truck
        vehicle_indices = np.where(
            (np.isin(classes, [2, 3, 5, 7])) & (confidences > self.conf_threshold)
        )[0]
        vehicles = boxes[vehicle_indices]

        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                iou, dist = self._calculate_iou_and_distance(vehicles[i], vehicles[j])
                if iou > self.crash_threshold or dist < self.min_distance:
                    return True
        return False

    def detect(self, video_path: str, save_path: str = "crash.jpg") -> tuple[str | None, bool]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file {video_path}")
            return None, False

        crash_frames = 0
        frame_count = 0
        self.crash_detected = False
        self.crash_frame = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            results = self.model.predict(
                source=frame, 
                conf=self.conf_threshold, 
                classes=[2, 3, 5, 7],  # Vehicle classes
                verbose=False
            )
            result = results[0]

            if len(result.boxes) == 0:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            if self.detect_crash_in_frame(boxes, classes, confidences):
                crash_frames += 1
                if crash_frames >= self.alert_count:
                    self.crash_detected = True
                    self.crash_frame = frame.copy()
                    cv2.putText(
                        self.crash_frame,
                        "CRASH DETECTED",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),
                        3,
                    )
                    break
            else:
                crash_frames = 0

        cap.release()
        return self._finalize_detection(save_path)

    def _finalize_detection(self, save_path: str) -> tuple[str | None, bool]:
        if self.crash_detected and self.crash_frame is not None:
            cv2.imwrite(save_path, self.crash_frame)
            print(f"Crash detected! Frame saved at: {save_path}")
            return save_path, True
        print("No crash detected.")
        return None, False

@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = f"{int(time.time())}_{file.filename}"
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(video_path)
    
    return jsonify({
        'message': 'Video uploaded successfully',
        'filename': filename
    }), 200

@app.route('/api/analyze-crash', methods=['POST'])
def analyze_crash():
    data = request.json
    if not data or 'filename' not in data:
        return jsonify({'error': 'No filename provided'}), 400
    
    video_path = os.path.join(UPLOAD_FOLDER, data['filename'])
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video file not found'}), 404
    
    result_filename = f"crash_result_{data['filename']}.jpg"
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    
    detector = CrashDetector()
    result, detected = detector.detect(video_path, result_path)
    
    if detected:
        return send_file(result_path, mimetype='image/jpeg')
    else:
        return jsonify({'message': 'No crash detected'}), 200

if __name__ == '__main__':
    app.run(port=5000, debug=True)