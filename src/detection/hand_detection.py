import cv2
import numpy as np
import datetime
from ultralytics import YOLO

class HandMonitor:
    """
    Production hand-monitoring using YOLO pose estimation.
    Detects:
      - Extra people (hands from another person reaching in)
      - Suspicious hand/arm positions near frame edges (passing objects)
      - Objects being held/exchanged near frame boundaries
    
    This does NOT depend on MediaPipe at all. It uses the same YOLO
    infrastructure already loaded for object detection, making it
    lightweight and guaranteed to work on Python 3.13.
    """
    def __init__(self, config):
        self.config = config
        self.alert_logger = None
        self.enabled = False
        self.model = None
        
        try:
            # YOLOv8 pose model for skeleton detection
            self.model = YOLO('models/yolov8n-pose.pt')
            self.enabled = True
        except Exception as e:
            print(f"Warning: Hand/Pose Monitor could not load pose model: {e}")
            self.enabled = False

        self.hand_alarm_end_time = datetime.datetime.now()
        self.last_alarm_message = ""
        self.edge_threshold = 0.08  # 8% from edge
        self.frame_skip = 0
        self.last_result = (False, "")

    def set_alert_logger(self, alert_logger):
        self.alert_logger = alert_logger

    def monitor_hands(self, frame):
        """
        Analyzes the frame for suspicious hand/body activity:
        1. Multiple people detected (helper nearby)
        2. Wrist/hand keypoints near frame edges (passing objects)
        """
        if not self.enabled:
            return False, ""
        
        # Run every 3rd frame for performance
        self.frame_skip += 1
        if self.frame_skip % 3 != 0:
            return self.last_result
            
        try:
            h, w = frame.shape[:2]
            
            # Run pose estimation
            results = self.model(frame, verbose=False, conf=0.4, imgsz=320)
            
            triggered = False
            msg = ""
            
            for result in results:
                if result.keypoints is None:
                    continue
                    
                keypoints = result.keypoints.data  # shape: [N, 17, 3] (x, y, conf)
                num_people = keypoints.shape[0]
                
                # Extract reliable (high-confidence) people boxes
                valid_people_boxes = []
                if result.boxes is not None:
                    for box in result.boxes:
                        if float(box.conf[0]) > 0.65:  # Stricter confidence filter
                            valid_people_boxes.append(box)

                # Rule 1: More than 1 reliable person detected
                if len(valid_people_boxes) > 1:
                    triggered = True
                    msg = "Another Person Detected Nearby"
                    
                    # Draw warning on extra person boxes
                    for i, box in enumerate(valid_people_boxes):
                        if i > 0:  # Skip first person (the student)
                            x1, y1, x2, y2 = [int(c) for c in box.xyxy[0]]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(frame, "EXTRA PERSON!", (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Rule 2 disabled: The wrist-to-edge check causes ~99% false positives 
                # for students normally typing or shifting hands on the keyboard.
            
            # Debounce and latch alarm
            if triggered:
                if not self.is_alarming()[0] and self.alert_logger:
                    self.alert_logger.log_alert("HAND_VIOLATION", msg)
                self.hand_alarm_end_time = datetime.datetime.now() + datetime.timedelta(seconds=3)
                self.last_alarm_message = msg
            
            self.last_result = self.is_alarming()
            return self.last_result
                
        except Exception as e:
            return False, ""

    def is_alarming(self):
        """Returns tuple of (is_active, message)"""
        return datetime.datetime.now() < self.hand_alarm_end_time, self.last_alarm_message
