import cv2
import numpy as np
from tensorflow.keras.models import load_model
from queue import Queue
from threading import Thread
import time

class VideoProcessor:
    def __init__(self, model_path):
        self.accident_model = load_model(model_path)
        self.object_detection_model = load_model('object_detection_model.h5')
        self.frame_queue = Queue(maxsize=32)
        self.stop_processing = False
        
    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            else:
                time.sleep(0.1)  # Prevent overloading
        cap.release()
        
    def detect_objects(self, frame):
        # Implement object detection (vehicles, pedestrians, etc.)
        resized_frame = cv2.resize(frame, (224, 224))
        normalized_frame = resized_frame / 255.0
        predictions = self.object_detection_model.predict(np.expand_dims(normalized_frame, axis=0))
        return predictions
        
    def detect_accident(self, frame, objects):
        # Extract features for accident detection
        features = self.extract_features(frame, objects)
        accident_prob = self.accident_model.predict(features)
        return accident_prob > 0.8  # Threshold
        
    def process_frames(self):
        while not self.stop_processing:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                objects = self.detect_objects(frame)
                is_accident = self.detect_accident(frame, objects)
                
                if is_accident:
                    self.trigger_alert(frame, objects)
                    
    def trigger_alert(self, frame, objects):
        # Save frame with bounding boxes
        timestamp = datetime.now().isoformat()
        alert_data = {
            "timestamp": timestamp,
            "location": "CAMERA_XYZ",  # Get from camera metadata
            "objects_involved": objects,
            "severity_score": self.calculate_severity(objects),
            "frame_path": f"alerts/{timestamp}.jpg"
        }
        # Save to database and trigger notifications
        save_alert_to_db(alert_data)
        notify_emergency_services(alert_data)
        
    def start_processing(self, video_path):
        extract_thread = Thread(target=self.extract_frames, args=(video_path,))
        process_thread = Thread(target=self.process_frames)
        
        extract_thread.start()
        process_thread.start()
        
        extract_thread.join()
        process_thread.join()