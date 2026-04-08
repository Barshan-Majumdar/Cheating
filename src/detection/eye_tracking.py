import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

class EyeTracker:
    def __init__(self, config):
        self.config = config
        self.enabled = False
        self.mp_face_mesh = None
        self.face_mesh = None
        
        try:
            # Setup MediaPipe face mesh cleanly
            self.mp_face_mesh = mp.solutions.face_mesh
            
            if self.mp_face_mesh:
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
                self.enabled = True
        except Exception as e:
            # Silently use MTCNN fallback for large scale
            self.enabled = False
        
        self.eye_alarm_end_time = datetime.now()
        
        self.config = config
        self.eye_threshold = config['detection']['eyes']['gaze_threshold']
        self.last_gaze_change = datetime.now()
        self.gaze_direction = "center"  # Default value
        self.eye_ratio = 0.3  # Default open eye ratio
        self.gaze_changes = 0
        self.alert_logger = None
        
        # Landmark indices for left and right eyes
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        
        # For EAR (Eye Aspect Ratio) calculation
        self.EYE_ASPECT_RATIO_THRESH = 0.3
        self.EYE_ASPECT_RATIO_CONSEC_FRAMES = 3

    def set_alert_logger(self, alert_logger):
        self.alert_logger = alert_logger

    def _calculate_ear(self, eye_points):
        # Compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear

    def track_eyes(self, frame, fallback_landmarks=None):
        if not self.enabled:
            # Fallback logic using MTCNN landmarks
            if fallback_landmarks is not None and len(fallback_landmarks) == 5:
                le, re, nose = fallback_landmarks[0], fallback_landmarks[1], fallback_landmarks[2]
                
                # Use euclidean distances for safety
                dist_left = np.linalg.norm(nose - le)
                dist_right = np.linalg.norm(nose - re)
                
                raw_ratio = (dist_left - dist_right) / (dist_left + dist_right + 1e-6)
                
                # EMA to smooth out MTCNN points jitter
                if not hasattr(self, 'smoothed_ratio'):
                    self.smoothed_ratio = raw_ratio
                    self.gaze_hold_frames = 0
                    self.last_stable_gaze = "center"
                else:
                    self.smoothed_ratio = 0.8 * self.smoothed_ratio + 0.2 * raw_ratio
                
                new_gaze = "center"
                if self.smoothed_ratio < -0.15:
                    new_gaze = "left"
                elif self.smoothed_ratio > 0.15:
                    new_gaze = "right"
                    
                # Require gaze to stabilize for a few frames
                if new_gaze != self.last_stable_gaze:
                    self.gaze_hold_frames += 1
                    if self.gaze_hold_frames >= 3:
                        self.last_stable_gaze = new_gaze
                        self.gaze_hold_frames = 0
                        
                        current_time = datetime.now()
                        if new_gaze != self.gaze_direction:
                            self.gaze_changes += 1
                            self.gaze_direction = new_gaze
                            self.last_gaze_change = current_time
                            
                        # Manage gaze change timer
                        if (self.gaze_changes > 3 and 
                            (current_time - self.last_gaze_change).total_seconds() < 2 and
                            self.alert_logger):
                            self.alert_logger.log_alert("EYE_MOVEMENT", "Excessive head/eye movement detected (Fallback module)")
                            self.eye_alarm_end_time = datetime.now() + __import__('datetime').timedelta(seconds=2)
                            self.gaze_changes = 0
                else:
                    self.gaze_hold_frames = 0
            
            return self.gaze_direction, self.eye_ratio
            
        try:
            # Convert frame to RGB and process
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if not results or not results.multi_face_landmarks:
                return self.gaze_direction, self.eye_ratio
            
            face_landmarks = results.multi_face_landmarks[0]
            frame_h, frame_w = frame.shape[:2]
            
            # Get eye landmarks in pixel coordinates
            left_eye_coords = np.array([(face_landmarks.landmark[i].x * frame_w, 
                                       face_landmarks.landmark[i].y * frame_h) 
                                      for i in self.LEFT_EYE_INDICES])
            
            right_eye_coords = np.array([(face_landmarks.landmark[i].x * frame_w, 
                                        face_landmarks.landmark[i].y * frame_h) 
                                       for i in self.RIGHT_EYE_INDICES])
            
            # Calculate Eye Aspect Ratio (EAR) for both eyes
            left_ear = self._calculate_ear(left_eye_coords)
            right_ear = self._calculate_ear(right_eye_coords)
            self.eye_ratio = (left_ear + right_ear) / 2.0
            
            # Calculate gaze direction based on eye position
            left_eye_center = np.mean(left_eye_coords, axis=0)
            right_eye_center = np.mean(right_eye_coords, axis=0)
            
            # Calculate horizontal difference between eye centers and nose
            nose_tip = np.array([face_landmarks.landmark[4].x * frame_w,
                                face_landmarks.landmark[4].y * frame_h])
            
            left_diff = left_eye_center[0] - nose_tip[0]
            right_diff = right_eye_center[0] - nose_tip[0]
            horiz_diff = (left_diff + right_diff) / 2.0
            
            # Determine gaze direction
            new_gaze = "center"
            if horiz_diff < -15:  # Looking left
                new_gaze = "left"
            elif horiz_diff > 15:  # Looking right
                new_gaze = "right"
            
            # Update gaze changes
            current_time = datetime.now()
            if new_gaze != self.gaze_direction:
                self.gaze_changes += 1
                self.gaze_direction = new_gaze
                self.last_gaze_change = current_time
                
            # Check for excessive eye movement
            if (self.gaze_changes > 3 and 
                (current_time - self.last_gaze_change).total_seconds() < 2 and
                self.alert_logger):
                self.alert_logger.log_alert(
                    "EYE_MOVEMENT",
                    "Excessive eye movement detected"
                )
                self.eye_alarm_end_time = datetime.now() + __import__('datetime').timedelta(seconds=2)
                self.gaze_changes = 0
            
            return self.gaze_direction, self.eye_ratio
            
        except Exception as e:
            if self.alert_logger:
                self.alert_logger.log_alert(
                    "EYE_TRACKING_ERROR",
                    f"Error in eye tracking: {str(e)}"
                )
            return self.gaze_direction, self.eye_ratio  # Return last known values

    def is_alarming(self):
        return datetime.now() < self.eye_alarm_end_time