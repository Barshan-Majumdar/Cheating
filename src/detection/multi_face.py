import cv2
import torch
from facenet_pytorch import MTCNN

class MultiFaceDetector:
    def __init__(self, config):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.detector = MTCNN(
            keep_all=True,
            post_process=False,
            min_face_size=80,
            thresholds=[0.7, 0.8, 0.8],
            device=self.device
        )
        self.threshold = config['detection']['multi_face']['alert_threshold']
        self.consecutive_frames = 0
        self.alert_logger = None
        self.frame_skip = 0
        self.last_result = False

    def set_alert_logger(self, alert_logger):
        self.alert_logger = alert_logger

    def detect_multiple_faces(self, frame):
        self.frame_skip += 1
        if self.frame_skip % 3 != 0:
            return self.last_result

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Prevent PyTorch from building massive computational graphs and OOM crashing
        with torch.no_grad():
            boxes, probs = self.detector.detect(rgb_frame)
        
        if boxes is not None and len(boxes) > 1:
            # Count faces with high confidence
            high_conf_faces = sum(p > 0.9 for p in probs)
            
            if high_conf_faces >= 2:
                self.consecutive_frames += 1
                if self.consecutive_frames >= self.threshold and self.alert_logger:
                    self.alert_logger.log_alert(
                        "MULTIPLE_FACES",
                        f"Detected {high_conf_faces} faces for {self.consecutive_frames} frames"
                    )
                    self.last_result = True
                    return True
            else:
                self.consecutive_frames = 0
        else:
            self.consecutive_frames = 0
            
        self.last_result = False
        return False