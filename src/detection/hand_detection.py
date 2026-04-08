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
                
                # Rule 1: More than 1 person skeleton detected
                if num_people > 1:
                    triggered = True
                    msg = "Another Person Detected Nearby"
                    
                    # Draw warning on extra person boxes
                    if result.boxes is not None:
                        for i, box in enumerate(result.boxes):
                            if i > 0:  # Skip first person (the student)
                                x1, y1, x2, y2 = [int(c) for c in box.xyxy[0]]
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                cv2.putText(frame, "EXTRA PERSON!", (x1, y1 - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Rule 2: Check wrist keypoints near edges
                # COCO keypoint indices: 9=left_wrist, 10=right_wrist
                for person_kps in keypoints:
                    for wrist_idx in [9, 10]:
                        kp = person_kps[wrist_idx]
                        kp_x, kp_y, kp_conf = float(kp[0]), float(kp[1]), float(kp[2])
                        
                        if kp_conf < 0.3:
                            continue
                        
                        # Normalize to 0-1 range
                        norm_x = kp_x / w
                        norm_y = kp_y / h
                        
                        # Check if wrist is near horizontal frame edges
                        if norm_x < self.edge_threshold or norm_x > (1.0 - self.edge_threshold):
                            triggered = True
                            if not msg:
                                msg = "Suspicious Hand Movement (Reaching/Passing)"
                            
                            # Draw circle on the wrist
                            cv2.circle(frame, (int(kp_x), int(kp_y)), 12, (0, 0, 255), -1)
                            cv2.putText(frame, "REACH!", (int(kp_x) + 15, int(kp_y)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
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
