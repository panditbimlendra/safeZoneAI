#!/usr/bin/env python3
"""
YOLOv11 Smart Surveillance System
Advanced surveillance with real-time analytics and alerts
"""

import cv2
import yaml
import json
import time
import threading
import logging
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
from ultralytics import YOLO
from ultralytics.solutions import heatmap
import torch
from typing import Dict, List, Tuple, Optional, Any

# Custom modules
from utils.camera_manager import CameraManager
from utils.alert_system import AlertSystem
from utils.database_handler import DatabaseHandler
from utils.visualization import Visualization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/surveillance.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SmartSurveillanceSystem:
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize the smart surveillance system"""
        self.config = self.load_config(config_path)
        self.running = False
        self.cameras = {}
        self.detections = defaultdict(list)
        self.tracks = defaultdict(dict)
        self.alerts = []
        self.analytics = {}
        
        # Initialize components
        self.init_components()
        logger.info("Smart Surveillance System initialized")
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def init_components(self):
        """Initialize all system components"""
        # Load YOLOv11 model
        self.model = YOLO(self.config['detection']['model'])
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize camera manager
        self.camera_manager = CameraManager(self.config['camera'])
        
        # Initialize alert system
        self.alert_system = AlertSystem(self.config['alert'])
        
        # Initialize database
        self.db = DatabaseHandler(self.config['storage']['database'])
        
        # Initialize visualization
        self.viz = Visualization()
        
        # Load zones
        self.zones = self.load_zones()
        
        # Initialize analytics trackers
        self.init_analytics()
        
        logger.info("All components initialized successfully")
    
    def load_zones(self) -> Dict:
        """Load surveillance zones from JSON file"""
        zones_file = Path("zones/zones.json")
        if zones_file.exists():
            with open(zones_file, 'r') as f:
                return json.load(f)
        return self.config.get('zones', {})
    
    def init_analytics(self):
        """Initialize analytics trackers"""
        self.analytics = {
            'person_count': 0,
            'vehicle_count': 0,
            'crowd_density': 0,
            'alerts_today': 0,
            'intrusions': [],
            'loitering_events': [],
            'abandoned_objects': [],
            'vehicle_violations': [],
            'heatmap_data': None,
            'movement_patterns': defaultdict(list)
        }
        
        # Object tracking history
        self.tracking_history = defaultdict(lambda: {
            'positions': deque(maxlen=100),
            'timestamps': deque(maxlen=100),
            'first_seen': None,
            'last_movement': None,
            'current_zone': None
        })
        
        # Abandoned object detection
        self.stationary_objects = defaultdict(lambda: {
            'first_detected': None,
            'last_position': None,
            'frames_stationary': 0
        })
        
        # Loitering detection
        self.loitering_timers = defaultdict(lambda: {
            'enter_time': None,
            'area_id': None,
            'alert_sent': False
        })
    
    def process_frame(self, camera_id: str, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with detection and analytics"""
        annotated_frame = frame.copy()
        
        # Run YOLOv11 inference
        results = self.model.track(
            frame, 
            persist=True,
            conf=self.config['detection']['confidence'],
            iou=self.config['detection']['iou'],
            classes=self.get_detection_classes(),
            verbose=False
        )
        
        if results and results[0].boxes:
            # Extract detections
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else None
            
            # Process each detection
            for i, box in enumerate(boxes):
                class_id = class_ids[i]
                confidence = confidences[i]
                track_id = track_ids[i] if track_ids is not None else i
                
                # Update tracking history
                self.update_tracking(camera_id, track_id, box, class_id)
                
                # Check for alerts and violations
                self.check_alerts(camera_id, track_id, box, class_id, frame)
                
                # Perform analytics
                self.update_analytics(camera_id, track_id, box, class_id)
                
                # Annotate frame
                annotated_frame = self.viz.draw_detection(
                    annotated_frame, box, class_id, confidence, track_id
                )
        
        # Add analytics overlays
        annotated_frame = self.add_analytics_overlay(annotated_frame, camera_id)
        
        # Draw zones
        annotated_frame = self.draw_zones(annotated_frame)
        
        # Update heatmap
        if self.config['analytics'].get('heatmap', {}).get('enabled', False):
            annotated_frame = self.update_heatmap(annotated_frame, results)
        
        return annotated_frame
    
    def get_detection_classes(self) -> List[int]:
        """Get list of classes to detect based on configuration"""
        classes = []
        detection_config = self.config['detection']['classes']
        
        if isinstance(detection_config, dict):
            # Add person class
            if 'person' in detection_config:
                classes.append(detection_config['person'])
            
            # Add vehicle classes
            if 'vehicle' in detection_config:
                if isinstance(detection_config['vehicle'], list):
                    classes.extend(detection_config['vehicle'])
                else:
                    classes.append(detection_config['vehicle'])
            
            # Add suspicious objects
            if 'suspicious_object' in detection_config:
                if isinstance(detection_config['suspicious_object'], list):
                    classes.extend(detection_config['suspicious_object'])
        
        return classes
    
    def update_tracking(self, camera_id: str, track_id: int, box: np.ndarray, class_id: int):
        """Update tracking history for an object"""
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        tracker_key = f"{camera_id}_{track_id}"
        history = self.tracking_history[tracker_key]
        
        current_time = datetime.now()
        
        # Update position history
        history['positions'].append((center_x, center_y))
        history['timestamps'].append(current_time)
        
        if history['first_seen'] is None:
            history['first_seen'] = current_time
        
        # Update current zone
        history['current_zone'] = self.get_zone_for_point(center_x, center_y)
        
        # Check if object moved
        if len(history['positions']) > 1:
            last_pos = history['positions'][-2]
            current_pos = history['positions'][-1]
            distance = np.sqrt((current_pos[0]-last_pos[0])**2 + (current_pos[1]-last_pos[1])**2)
            
            if distance > 5:  # Movement threshold in pixels
                history['last_movement'] = current_time
                
                # Reset stationary counter if object moved
                if tracker_key in self.stationary_objects:
                    del self.stationary_objects[tracker_key]
    
    def check_alerts(self, camera_id: str, track_id: int, box: np.ndarray, class_id: int, frame: np.ndarray):
        """Check for various alert conditions"""
        tracker_key = f"{camera_id}_{track_id}"
        history = self.tracking_history[tracker_key]
        current_time = datetime.now()
        
        # 1. Intrusion Detection (Restricted Zones)
        if self.config['analytics']['intrusion']['enabled']:
            self.check_intrusion(camera_id, track_id, history, frame)
        
        # 2. Loitering Detection
        if self.config['analytics']['loitering']['enabled']:
            self.check_loitering(camera_id, track_id, history, frame)
        
        # 3. Abandoned Object Detection
        if self.config['analytics']['abandoned_object']['enabled']:
            self.check_abandoned_object(camera_id, track_id, box, class_id, frame)
        
        # 4. Crowd Density Alert
        if self.config['analytics']['crowd_density']['enabled']:
            self.check_crowd_density(camera_id, frame)
        
        # 5. Vehicle Violations
        if self.config['analytics']['vehicle_analytics']['enabled'] and class_id in [1, 2, 3, 5, 7]:
            self.check_vehicle_violations(camera_id, track_id, box, class_id, history, frame)
    
    def check_intrusion(self, camera_id: str, track_id: int, history: Dict, frame: np.ndarray):
        """Check for intrusion in restricted zones"""
        if history['current_zone'] and history['current_zone']['type'] == 'restricted':
            # Check if alert already sent for this object in this zone
            alert_key = f"intrusion_{camera_id}_{track_id}_{history['current_zone']['name']}"
            
            if not any(alert['key'] == alert_key for alert in self.alerts):
                # Wait for alarm delay
                enter_time = history['timestamps'][-1]
                if (datetime.now() - enter_time).seconds >= self.config['analytics']['intrusion'].get('alarm_delay', 3):
                    # Send intrusion alert
                    alert = {
                        'type': 'intrusion',
                        'camera': camera_id,
                        'track_id': track_id,
                        'zone': history['current_zone']['name'],
                        'timestamp': datetime.now(),
                        'severity': 'high',
                        'key': alert_key
                    }
                    
                    self.alerts.append(alert)
                    self.analytics['intrusions'].append(alert)
                    
                    # Send alert notification
                    self.send_alert(alert, frame)
                    logger.warning(f"Intrusion alert: {alert}")
    
    def check_loitering(self, camera_id: str, track_id: int, history: Dict, frame: np.ndarray):
        """Check for loitering behavior"""
        tracker_key = f"{camera_id}_{track_id}"
        
        if history['first_seen']:
            time_in_area = (datetime.now() - history['first_seen']).seconds
            
            if time_in_area > self.config['analytics']['loitering']['threshold_time']:
                # Check if area is too small (indicating limited movement)
                if len(history['positions']) > 10:
                    positions = np.array(history['positions'])
                    area = self.calculate_movement_area(positions)
                    
                    if area < self.config['analytics']['loitering']['threshold_area']:
                        # Loitering detected
                        alert_key = f"loitering_{tracker_key}"
                        
                        if not self.loitering_timers[tracker_key]['alert_sent']:
                            alert = {
                                'type': 'loitering',
                                'camera': camera_id,
                                'track_id': track_id,
                                'duration': time_in_area,
                                'timestamp': datetime.now(),
                                'severity': 'medium',
                                'key': alert_key
                            }
                            
                            self.alerts.append(alert)
                            self.analytics['loitering_events'].append(alert)
                            self.loitering_timers[tracker_key]['alert_sent'] = True
                            
                            # Send alert
                            self.send_alert(alert, frame)
                            logger.warning(f"Loitering alert: {alert}")
    
    def check_abandoned_object(self, camera_id: str, track_id: int, box: np.ndarray, 
                              class_id: int, frame: np.ndarray):
        """Check for abandoned/stationary objects"""
        if class_id in [26, 28, 39]:  # Backpack, suitcase, bottle
            tracker_key = f"{camera_id}_{track_id}"
            
            if tracker_key not in self.stationary_objects:
                self.stationary_objects[tracker_key] = {
                    'first_detected': datetime.now(),
                    'last_position': box,
                    'frames_stationary': 0
                }
            else:
                # Check if object is stationary
                last_box = self.stationary_objects[tracker_key]['last_position']
                iou = self.calculate_iou(box, last_box)
                
                if iou > 0.7:  # Object hasn't moved much
                    self.stationary_objects[tracker_key]['frames_stationary'] += 1
                    
                    stationary_time = (datetime.now() - 
                                     self.stationary_objects[tracker_key]['first_detected']).seconds
                    
                    if stationary_time > self.config['analytics']['abandoned_object']['stationary_time']:
                        # Abandoned object detected
                        alert_key = f"abandoned_{tracker_key}"
                        
                        if not any(alert['key'] == alert_key for alert in self.alerts):
                            alert = {
                                'type': 'abandoned_object',
                                'camera': camera_id,
                                'track_id': track_id,
                                'object_class': class_id,
                                'duration': stationary_time,
                                'timestamp': datetime.now(),
                                'severity': 'high',
                                'key': alert_key
                            }
                            
                            self.alerts.append(alert)
                            self.analytics['abandoned_objects'].append(alert)
                            
                            # Send alert
                            self.send_alert(alert, frame)
                            logger.warning(f"Abandoned object alert: {alert}")
                else:
                    # Object moved, reset counter
                    self.stationary_objects[tracker_key] = {
                        'first_detected': datetime.now(),
                        'last_position': box,
                        'frames_stationary': 0
                    }
    
    def check_crowd_density(self, camera_id: str, frame: np.ndarray):
        """Check for high crowd density"""
        person_count = sum(1 for track in self.tracking_history.values() 
                          if track.get('class_id') == 0)  # Person class
        
        self.analytics['person_count'] = person_count
        
        if person_count >= self.config['analytics']['crowd_density']['critical_threshold']:
            alert_key = f"crowd_{camera_id}_{datetime.now().strftime('%H%M')}"
            
            if not any(alert['key'] == alert_key for alert in self.alerts):
                alert = {
                    'type': 'crowd_density',
                    'camera': camera_id,
                    'person_count': person_count,
                    'threshold': self.config['analytics']['crowd_density']['critical_threshold'],
                    'timestamp': datetime.now(),
                    'severity': 'medium',
                    'key': alert_key
                }
                
                self.alerts.append(alert)
                
                # Send alert
                self.send_alert(alert, frame)
                logger.warning(f"Crowd density alert: {alert}")
    
    def check_vehicle_violations(self, camera_id: str, track_id: int, box: np.ndarray, 
                                class_id: int, history: Dict, frame: np.ndarray):
        """Check for vehicle violations"""
        # Calculate speed if enough history
        if len(history['positions']) >= 10:
            speed = self.calculate_speed(history)
            
            if speed > self.config['analytics']['vehicle_analytics']['speed_limit']:
                # Speeding violation
                alert = {
                    'type': 'speeding',
                    'camera': camera_id,
                    'track_id': track_id,
                    'speed': speed,
                    'limit': self.config['analytics']['vehicle_analytics']['speed_limit'],
                    'timestamp': datetime.now(),
                    'severity': 'medium'
                }
                
                self.alerts.append(alert)
                self.analytics['vehicle_violations'].append(alert)
                
                # Send alert
                self.send_alert(alert, frame)
                logger.warning(f"Speeding alert: {alert}")
        
        # Check parking violation
        if history['first_seen']:
            parking_time = (datetime.now() - history['first_seen']).seconds
            
            if parking_time > self.config['analytics']['vehicle_analytics']['parking_time_limit']:
                # Parking violation
                alert_key = f"parking_{camera_id}_{track_id}"
                
                if not any(alert['key'] == alert_key for alert in self.alerts):
                    alert = {
                        'type': 'parking_violation',
                        'camera': camera_id,
                        'track_id': track_id,
                        'duration': parking_time,
                        'limit': self.config['analytics']['vehicle_analytics']['parking_time_limit'],
                        'timestamp': datetime.now(),
                        'severity': 'low',
                        'key': alert_key
                    }
                    
                    self.alerts.append(alert)
                    self.analytics['vehicle_violations'].append(alert)
                    
                    # Send alert
                    self.send_alert(alert, frame)
                    logger.warning(f"Parking violation alert: {alert}")
    
    def calculate_speed(self, history: Dict) -> float:
        """Calculate object speed in pixels/second"""
        if len(history['positions']) < 2 or len(history['timestamps']) < 2:
            return 0.0
        
        positions = np.array(history['positions'])
        timestamps = np.array([ts.timestamp() for ts in history['timestamps']])
        
        # Calculate total distance and time
        total_distance = 0
        for i in range(1, len(positions)):
            p1 = positions[i-1]
            p2 = positions[i]
            total_distance += np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        
        total_time = timestamps[-1] - timestamps[0]
        
        if total_time > 0:
            speed_px_per_sec = total_distance / total_time
            # Convert to km/h (requires calibration)
            return speed_px_per_sec * 0.1  # Approximate conversion
        
        return 0.0
    
    def calculate_movement_area(self, positions: np.ndarray) -> float:
        """Calculate area covered by object movement"""
        if len(positions) < 2:
            return 0
        
        min_x, min_y = np.min(positions, axis=0)
        max_x, max_y = np.max(positions, axis=0)
        
        area = (max_x - min_x) * (max_y - min_y)
        return area
    
    def calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate Intersection over Union for two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def get_zone_for_point(self, x: float, y: float) -> Optional[Dict]:
        """Get zone for a given point"""
        for zone_type, zones in self.zones.items():
            for zone in zones:
                if self.is_point_in_polygon((x, y), zone['polygon']):
                    return {'type': zone_type[:-1], 'name': zone['name'], 'polygon': zone['polygon']}
        return None
    
    def is_point_in_polygon(self, point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        """Check if point is inside polygon using ray casting algorithm"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def send_alert(self, alert: Dict, frame: np.ndarray):
        """Send alert through configured channels"""
        # Save snapshot if enabled
        if self.config['storage']['snapshots']['save_on_alert']:
            snapshot_path = self.save_snapshot(alert, frame)
            alert['snapshot_path'] = snapshot_path
        
        # Send via alert system
        self.alert_system.send(alert)
        
        # Log to database
        self.db.log_alert(alert)
        
        # Increment alert counter
        self.analytics['alerts_today'] += 1
    
    def save_snapshot(self, alert: Dict, frame: np.ndarray) -> str:
        """Save snapshot of alert frame"""
        timestamp = alert['timestamp'].strftime("%Y%m%d_%H%M%S")
        filename = f"{alert['type']}_{alert['camera']}_{timestamp}.jpg"
        snapshot_dir = Path(self.config['storage']['snapshots']['path'])
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        snapshot_path = snapshot_dir / filename
        cv2.imwrite(str(snapshot_path), frame)
        
        return str(snapshot_path)
    
    def add_analytics_overlay(self, frame: np.ndarray, camera_id: str) -> np.ndarray:
        """Add analytics information overlay to frame"""
        overlay = frame.copy()
        
        # Add analytics text
        texts = [
            f"Camera: {camera_id}",
            f"Persons: {self.analytics['person_count']}",
            f"Vehicles: {self.analytics['vehicle_count']}",
            f"Alerts: {self.analytics['alerts_today']}",
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        ]
        
        for i, text in enumerate(texts):
            cv2.putText(overlay, text, (10, 30 + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add zone information
        for zone_type, zones in self.zones.items():
            for zone in zones:
                color = (0, 0, 255) if zone_type == 'restricted_zones' else (0, 255, 0)
                points = np.array(zone['polygon'], np.int32)
                cv2.polylines(overlay, [points], True, color, 2)
                # Add zone label
                centroid = np.mean(points, axis=0).astype(int)
                cv2.putText(overlay, zone['name'], tuple(centroid),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Blend overlay
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
        
        return frame
    
    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw surveillance zones on frame"""
        for zone_type, zones in self.zones.items():
            for zone in zones:
                color = (0, 0, 255) if zone_type == 'restricted_zones' else (0, 255, 0)
                alpha = 0.3
                
                # Create overlay
                overlay = frame.copy()
                points = np.array(zone['polygon'], np.int32)
                cv2.fillPoly(overlay, [points], color)
                
                # Blend
                frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
                
                # Draw border
                cv2.polylines(frame, [points], True, color, 2)
        
        return frame
    
    def update_heatmap(self, frame: np.ndarray, results) -> np.ndarray:
        """Update and display heatmap"""
        # Initialize heatmap if not exists
        if self.analytics['heatmap_data'] is None:
            self.analytics['heatmap_data'] = heatmap.Heatmap(
                colormap=cv2.COLORMAP_JET,
                view_img=True,
                shape="circle"
            )
        
        # Update heatmap with detections
        if results and results[0].boxes:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for box in boxes:
                x_center = int((box[0] + box[2]) / 2)
                y_center = int((box[1] + box[3]) / 2)
                self.analytics['heatmap_data'].generate_heatmap(frame, [x_center, y_center])
        
        return frame
    
    def start_camera_stream(self, camera_id: str):
        """Start processing stream for a specific camera"""
        camera = self.camera_manager.get_camera(camera_id)
        if not camera:
            logger.error(f"Camera {camera_id} not found")
            return
        
        logger.info(f"Starting stream for camera: {camera_id}")
        
        while self.running:
            ret, frame = camera.read()
            if not ret:
                logger.warning(f"Camera {camera_id} stream interrupted")
                time.sleep(1)
                continue
            
            # Process frame
            processed_frame = self.process_frame(camera_id, frame)
            
            # Display frame
            cv2.imshow(f"Camera {camera_id}", processed_frame)
            
            # Save frame if recording
            if camera['recording']:
                camera['video_writer'].write(processed_frame)
            
            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        camera.release()
        cv2.destroyWindow(f"Camera {camera_id}")
    
    def start(self):
        """Start the surveillance system"""
        self.running = True
        logger.info("Starting Smart Surveillance System")
        
        # Start camera threads
        threads = []
        for camera_id in self.camera_manager.get_camera_ids():
            thread = threading.Thread(
                target=self.start_camera_stream,
                args=(camera_id,),
                daemon=True
            )
            threads.append(thread)
            thread.start()
        
        # Start dashboard server
        dashboard_thread = threading.Thread(
            target=self.start_dashboard,
            daemon=True
        )
        dashboard_thread.start()
        
        logger.info(f"Started {len(threads)} camera threads")
        
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1)
                # Cleanup old alerts
                self.cleanup_old_alerts()
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            self.stop()
    
    def start_dashboard(self):
        """Start the web dashboard"""
        from dashboard import start_dashboard
        start_dashboard(self)
    
    def cleanup_old_alerts(self):
        """Remove old alerts from memory"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.alerts = [alert for alert in self.alerts if alert['timestamp'] > cutoff_time]
    
    def stop(self):
        """Stop the surveillance system"""
        self.running = False
        logger.info("Stopping Smart Surveillance System")
        
        # Stop all cameras
        self.camera_manager.stop_all()
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        
        # Save analytics data
        self.save_analytics()
        
        logger.info("System stopped successfully")
    
    def save_analytics(self):
        """Save analytics data to file"""
        analytics_file = Path("logs/analytics.json")
        with open(analytics_file, 'w') as f:
            json.dump(self.analytics, f, default=str, indent=2)
        
        logger.info(f"Analytics saved to {analytics_file}")
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            'running': self.running,
            'cameras_active': len(self.camera_manager.get_active_cameras()),
            'total_detections': len(self.tracking_history),
            'active_alerts': len([a for a in self.alerts 
                                 if (datetime.now() - a['timestamp']).seconds < 300]),
            'analytics': self.analytics
        }


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv11 Smart Surveillance System")
    parser.add_argument("--config", default="config/settings.yaml", help="Configuration file path")
    parser.add_argument("--camera", help="Process specific camera only")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable web dashboard")
    
    args = parser.parse_args()
    
    # Initialize system
    system = SmartSurveillanceSystem(args.config)
    
    # Start system
    try:
        system.start()
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
    finally:
        system.stop()


if __name__ == "__main__":
    main()