import cv2
import mediapipe as mp
import numpy as np

class MouthMonitor:
    def __init__(self, config):
        try:
            # Setup MediaPipe face mesh cleanly
            self.mp_face_mesh = mp.solutions.face_mesh
        except Exception as e:
            # Silently use MTCNN fallback for large scale
            self.mp_face_mesh = None
            
        self.mouth_alarm_end_time = __import__('datetime').datetime.now()
            
        if self.mp_face_mesh:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)
        else:
            self.face_mesh = None
            
        self.mouth_threshold = config['detection']['mouth']['movement_threshold']
        self.mouth_movement_count = 0
        self.last_mouth_time = None
        self.alert_logger = None  # Will be set externally
        
    def set_alert_logger(self, alert_logger):
        self.alert_logger = alert_logger
        
    def monitor_mouth(self, frame, fallback_landmarks=None):
        if not self.face_mesh:
            if fallback_landmarks is not None and len(fallback_landmarks) == 5:
                # Use mouth corners (indices 3 and 4)
                lmc, rmc = fallback_landmarks[3], fallback_landmarks[4]
                width = np.linalg.norm(lmc - rmc)
                
                nose = fallback_landmarks[2]
                mouth_center = (lmc + rmc) / 2.0
                vert_dist = np.linalg.norm(nose - mouth_center)
                
                le, re = fallback_landmarks[0], fallback_landmarks[1]
                eye_dist = np.linalg.norm(le - re) + 1e-6
                
                # Metric incorporating absolute width and vertical drop
                metric = (width + vert_dist) / eye_dist
                
                if not hasattr(self, 'smoothed_metric'):
                    self.smoothed_metric = metric
                    return False  # Skip first frame entirely
                    
                diff = abs(metric - self.smoothed_metric)
                self.smoothed_metric = 0.9 * self.smoothed_metric + 0.1 * metric
                
                if diff > 0.08:
                    self.mouth_movement_count += 1
                    if self.mouth_movement_count > self.mouth_threshold and self.alert_logger:
                        self.alert_logger.log_alert("MOUTH_MOVEMENT", "Excessive mouth movement detected (Fallback module)")
                        self.mouth_alarm_end_time = __import__('datetime').datetime.now() + __import__('datetime').timedelta(seconds=2)
                        self.mouth_movement_count = 0
                    return True
                else:
                    self.mouth_movement_count = max(0, self.mouth_movement_count - 2)
                    
            return False
            
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return False
            
        face_landmarks = results.multi_face_landmarks[0]
        
        # Get mouth landmarks (using more points for better accuracy)
        mouth_points = [
            13,  # Upper inner lip
            14,  # Lower inner lip
            78,  # Right corner
            306,  # Left corner
            312,  # Upper outer lip
            317,  # Lower outer lip
        ]
        
        # Calculate mouth openness
        upper_lip = face_landmarks.landmark[13].y
        lower_lip = face_landmarks.landmark[14].y
        mouth_open = lower_lip - upper_lip
        
        # Calculate mouth width
        right_corner = face_landmarks.landmark[78].x
        left_corner = face_landmarks.landmark[306].x
        mouth_width = abs(right_corner - left_corner)
        
        if mouth_open > 0.03 or mouth_width > 0.2:  # Thresholds for mouth movement
            self.mouth_movement_count += 1
            
            if self.mouth_movement_count > self.mouth_threshold and self.alert_logger:
                self.alert_logger.log_alert(
                    "MOUTH_MOVEMENT", 
                    "Excessive mouth movement detected (possible talking)"
                )
                self.mouth_alarm_end_time = __import__('datetime').datetime.now() + __import__('datetime').timedelta(seconds=2)
                self.mouth_movement_count = 0
            return True
        else:
            self.mouth_movement_count = max(0, self.mouth_movement_count - 1)
            return False

    def is_alarming(self):
        return __import__('datetime').datetime.now() < self.mouth_alarm_end_time