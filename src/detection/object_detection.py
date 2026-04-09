import cv2
import torch
from ultralytics import YOLO
from datetime import datetime, timedelta

class ObjectDetector:
    def __init__(self, config):
        self.config = config['detection']['objects']
        self.model = None
        # Expanded suspicious object map from COCO dataset
        self.class_map = {
            0:  'person',       # Extra person in frame
            24: 'backpack',
            25: 'umbrella',
            26: 'handbag',
            39: 'bottle',
            63: 'laptop',
            65: 'remote',
            66: 'keyboard',
            67: 'cell phone',
            73: 'book',
            74: 'clock',
            76: 'scissors',
        }
        # These are the ones we actually flag as cheating (not person)
        self.forbidden_classes = {24, 25, 26, 63, 65, 66, 67, 73, 74, 76}
        
        self.alert_logger = None
        self.object_alarm_end_time = datetime.now()
        self.last_detected_label = ""
        self.detection_interval = self.config['detection_interval']
        self.frame_count = 0
        self.last_detected = False
        self.last_person_detected = False
        self._initialize_model()
        self.last_detection_time = datetime.now()

        # ── Real Model Metrics Counters ────────────────────────────────
        self.metrics = {
            'inference_frames': 0,       # Total frames where YOLO ran
            'raw_detections': 0,         # All YOLO outputs above conf threshold
            'validated_detections': 0,   # Passed geometric filters (model TP)
            'rejected_detections': 0,    # Failed geometric filters (model FP)
            'no_detection_frames': 0,    # Frames with zero forbidden objects
            'confidences': [],           # Confidence of every validated detection
        }

    def _initialize_model(self):
        """Initialize optimized YOLO model"""
        try:
            # Use YOLOv8s model for better accuracy than nano to prevent false positives
            self.model = YOLO('models/yolov8s.pt')
            
            # Optimize model settings
            self.model.overrides['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.overrides['imgsz'] = 480  # Bigger for better small-object detection
            self.model.overrides['iou'] = 0.45
            
            # Warm up the model
            dummy_input = torch.zeros((1, 3, 480, 480)).to(self.model.device)
            self.model(dummy_input)
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize object detector: {str(e)}")

    def set_alert_logger(self, alert_logger):
        self.alert_logger = alert_logger

    def detect_objects(self, frame):
        """Object detection with bounding box visualization always on"""
        current_time = datetime.now()
        time_since_last = (current_time - self.last_detection_time).total_seconds()
        
        # Skip detection if not enough time has passed
        if time_since_last < (1.0 / self.config['max_fps']):
            return self.last_detected, self.last_person_detected
            
        try:
            orig_h, orig_w = frame.shape[:2]
            new_w = 480
            new_h = int(orig_h * (new_w / orig_w))
            resized_frame = cv2.resize(frame, (new_w, new_h))
            
            min_conf = self.config.get('min_confidence', 0.65)
            # Run inference with confidence from config to filter false positives
            results = self.model(resized_frame, verbose=False, conf=min_conf)
            
            detected = False
            person_detected = False
            detected_labels = []
            frame_raw = 0       # raw YOLO forbidden-object detections this frame
            frame_validated = 0 # detections that pass all filters this frame
            frame_rejected = 0  # detections rejected by geometric filters
            
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    
                    # Person tracking for occlusion fallback
                    if cls == 0 and conf > 0.5:
                        person_detected = True

                    # Forbidden object detection
                    if cls in self.forbidden_classes and conf > min_conf:
                        frame_raw += 1
                        
                        # Apply custom labeling logic for cell phones (cls=67) to handle false positives
                        if cls == 67:
                            x1_tmp, y1_tmp, x2_tmp, y2_tmp = box.xyxy[0]
                            box_w = float(x2_tmp - x1_tmp)
                            box_h = float(y2_tmp - y1_tmp)
                            
                            if box_w > 0 and box_h > 0:
                                aspect_ratio = max(box_w, box_h) / min(box_w, box_h)
                                area_ratio = (box_w * box_h) / (new_w * new_h)
                                
                                # Reject extremely tall/thin objects (like perfume bottles aspect ratio > 2.8)
                                # Reject impossibly tiny (<0.5%) or monstrously large (>60%) detections
                                if aspect_ratio > 2.8 or area_ratio < 0.005 or area_ratio > 0.60:
                                    frame_rejected += 1
                                    continue
                                    
                            # Since realistic webcams max out at ~88% confidence for real phones,
                            # 80% is a safe and highly accurate threshold now that geometric shapes filter out bottles.
                            if conf > 0.80:
                                label = "cell phone"
                            else:
                                label = "unidentified object"
                        else:
                            label = self.class_map.get(cls, f"object_{cls}")
                            
                        detected = True
                        detected_labels.append(label)
                        frame_validated += 1
                        self.metrics['confidences'].append(conf)
                        
                        # Always draw bounding boxes on forbidden objects
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1 = int(x1 * (orig_w / new_w))
                        y1 = int(y1 * (orig_h / new_h))
                        x2 = int(x2 * (orig_w / new_w))
                        y2 = int(y2 * (orig_h / new_h))
                        
                        # Red bounding box with label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        text = f"ALERT: {label.upper()} ({conf:.0%})"
                        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), (0, 0, 255), -1)
                        cv2.putText(frame, text, (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if detected and detected_labels:
                combined_label = ", ".join(set(detected_labels))
                if self.alert_logger:
                    self.alert_logger.log_alert(
                        "OBJECT_DETECTED",
                        f"Unauthorized object detected: {combined_label}"
                    )
                self.object_alarm_end_time = datetime.now() + timedelta(seconds=3)
                self.last_detected_label = combined_label
            
            # ── Update real metrics ──────────────────────────────────
            self.metrics['inference_frames'] += 1
            self.metrics['raw_detections'] += frame_raw
            self.metrics['validated_detections'] += frame_validated
            self.metrics['rejected_detections'] += frame_rejected
            if frame_raw == 0:
                self.metrics['no_detection_frames'] += 1

            self.last_detection_time = current_time
            self.last_detected = detected
            self.last_person_detected = person_detected
            return detected, person_detected
            
        except Exception as e:
            if self.alert_logger:
                self.alert_logger.log_alert(
                    "OBJECT_DETECTION_ERROR",
                    f"Object detection failed: {str(e)}"
                )
            return self.last_detected, self.last_person_detected

    def is_alarming(self):
        return datetime.now() < self.object_alarm_end_time, self.last_detected_label