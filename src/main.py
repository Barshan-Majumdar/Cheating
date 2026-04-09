import cv2
import yaml
import math
import time
import os
import threading
from datetime import datetime
from detection.face_detection import FaceDetector
from detection.eye_tracking import EyeTracker
from detection.mouth_detection import MouthMonitor
from detection.object_detection import ObjectDetector
from detection.multi_face import MultiFaceDetector
from detection.audio_detection import AudioMonitor
from utils.video_utils import VideoRecorder
from utils.screen_capture import ScreenRecorder
from utils.logging import AlertLogger
from utils.alert_system import AlertSystem
from utils.violation_logger import ViolationLogger
from utils.screenshot_utils import ViolationCapturer
from reporting.report_generator import ReportGenerator
from utils.hardware_checks import HardwareMonitor


def load_config():
    with open('config/config.yaml') as f:
        return yaml.safe_load(f)


# ══════════════════════════════════════════════════════════════════════════
#  METRICS TRACKER — Real ML Model Metrics with Ground-Truth Labeling
# ══════════════════════════════════════════════════════════════════════════

class MetricsTracker:
    """
    Computes genuine classification metrics by comparing the system's
    predictions against manual ground-truth labels.

    How it works:
      • The user presses 'v' to toggle ground-truth state:
        - When 'v' is pressed → "a violation IS happening" (ground truth = positive)
        - Press 'v' again   → "violation stopped" (ground truth = negative)
      • Each frame is recorded with BOTH:
        - predicted: did the system detect a violation?
        - actual:    was the user marking a ground-truth violation?

    Confusion matrix (genuine):
      TP = system detected violation AND user confirmed it was real
      FP = system detected violation BUT user said no violation
      TN = system said clean AND user confirmed no violation
      FN = system said clean BUT user said violation was happening

    All metrics are derived from these real comparisons.
    """

    def __init__(self):
        self.start_time = time.time()
        self.total_frames = 0
        self.tp = 0   # system detected + user confirmed
        self.fp = 0   # system detected + user said no
        self.tn = 0   # system clean + user confirmed clean
        self.fn = 0   # system clean + user said violation
        self.ground_truth_active = False  # toggled by 'v' key
        self.per_detector_triggers = {}
        self.obj_confidences = []   # actual YOLO confidence scores
        self.face_confidences = []  # actual MTCNN confidence scores

    def toggle_ground_truth(self):
        """Called when user presses 'v' — toggle ground-truth violation state."""
        self.ground_truth_active = not self.ground_truth_active
        state = "VIOLATION ACTIVE" if self.ground_truth_active else "CLEAN"
        print(f"[GT] Ground truth: {state}")

    def record_frame(self, predicted_violation: bool, active_detectors: list = None):
        """
        Record one frame with the system's prediction vs ground truth.
        Called every frame from the main loop.
        """
        self.total_frames += 1
        actual = self.ground_truth_active

        if predicted_violation and actual:
            self.tp += 1
        elif predicted_violation and not actual:
            self.fp += 1
        elif not predicted_violation and not actual:
            self.tn += 1
        elif not predicted_violation and actual:
            self.fn += 1

        if active_detectors:
            for name in active_detectors:
                self.per_detector_triggers[name] = self.per_detector_triggers.get(name, 0) + 1

    def print_report(self, detectors: list):
        """
        Print a full metrics report using:
        - TP/FP/TN/FN from prediction vs ground-truth comparison
        - Confidence scores from actual model internals
        """
        elapsed = time.time() - self.start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        avg_fps = round(self.total_frames / elapsed, 1) if elapsed > 0 else 0

        tp, fp, tn, fn = self.tp, self.fp, self.tn, self.fn
        total = tp + fp + tn + fn
        if total == 0:
            total = max(self.total_frames, 1)
            tn = total

        # ── Compute the 5 metrics from REAL confusion matrix ───────────
        accuracy    = (tp + tn) / total
        precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1          = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Matthews Correlation Coefficient
        denom_sq = (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)
        mcc = ((tp * tn) - (fp * fn)) / math.sqrt(denom_sq) if denom_sq > 0 else 0.0

        # ── Collect actual confidence data from detector objects ────────
        obj_det = next((d for d in detectors if type(d).__name__ == "ObjectDetector"), None)
        face_det = next((d for d in detectors if type(d).__name__ == "FaceDetector"), None)

        obj_m = obj_det.metrics if obj_det else {}
        obj_inference = obj_m.get('inference_frames', 0)
        obj_raw = obj_m.get('raw_detections', 0)
        obj_validated = obj_m.get('validated_detections', 0)
        obj_rejected = obj_m.get('rejected_detections', 0)
        obj_confs = obj_m.get('confidences', [])
        avg_obj_conf = sum(obj_confs) / len(obj_confs) if obj_confs else 0.0
        max_obj_conf = max(obj_confs) if obj_confs else 0.0
        min_obj_conf = min(obj_confs) if obj_confs else 0.0

        face_m = face_det.metrics if face_det else {}
        face_inf = face_m.get('inference_frames', 0)
        face_detected = face_m.get('face_detected_frames', 0)
        face_absent = face_m.get('face_absent_frames', 0)
        face_violations = face_m.get('violation_frames', 0)
        face_confs = face_m.get('confidences', [])
        avg_face_conf = sum(face_confs) / len(face_confs) if face_confs else 0.0

        # ── Print Report ───────────────────────────────────────────────
        gt_used = (tp + fp + fn) > 0 or self.ground_truth_active
        note = "" if gt_used else "  ⚠  No ground-truth was marked (press 'v' during session)"

        print("\n")
        print("=" * 66)
        print("       PROCTORING SESSION — ML MODEL PERFORMANCE METRICS")
        print("=" * 66)
        if note:
            print(note)

        print(f"\n  Session Duration    : {mins}m {secs}s")
        print(f"  Total Frames        : {self.total_frames}")
        print(f"  Average FPS         : {avg_fps}")

        print(f"\n  ┌───────────────────────────────────────────────────┐")
        print(f"  │  CONFUSION MATRIX  (prediction vs ground truth)   │")
        print(f"  ├─────────────────┬───────────────┬─────────────────┤")
        print(f"  │                 │ Actual (+)    │ Actual (-)      │")
        print(f"  ├─────────────────┼───────────────┼─────────────────┤")
        print(f"  │ Predicted (+)   │ TP = {tp:<8} │ FP = {fp:<10} │")
        print(f"  │ Predicted (-)   │ FN = {fn:<8} │ TN = {tn:<10} │")
        print(f"  └─────────────────┴───────────────┴─────────────────┘")

        print(f"\n  ┌───────────────────────────────────────────────────┐")
        print(f"  │  CLASSIFICATION METRICS                           │")
        print(f"  ├────────────────────────┬──────────────────────────┤")
        print(f"  │  1. Accuracy           │  {accuracy:.4f}  ({accuracy*100:.1f}%)          │")
        print(f"  │  2. Precision          │  {precision:.4f}  ({precision*100:.1f}%)          │")
        print(f"  │  3. Specificity (TNR)  │  {specificity:.4f}  ({specificity*100:.1f}%)          │")
        print(f"  │  4. F1 Score           │  {f1:.4f}  ({f1*100:.1f}%)          │")
        print(f"  │  5. MCC                │  {mcc:+.4f}                   │")
        print(f"  ├────────────────────────┼──────────────────────────┤")
        print(f"  │  Recall (Sensitivity)  │  {recall:.4f}  ({recall*100:.1f}%)          │")
        print(f"  └────────────────────────┴──────────────────────────┘")

        print(f"\n  ┌───────────────────────────────────────────────────┐")
        print(f"  │  YOLO OBJECT DETECTOR (yolov8s.pt)                │")
        print(f"  ├────────────────────────┬──────────────────────────┤")
        print(f"  │  Inference Frames      │  {obj_inference:<24} │")
        print(f"  │  Raw Detections        │  {obj_raw:<24} │")
        print(f"  │  Validated             │  {obj_validated:<24} │")
        print(f"  │  Rejected (geo filter) │  {obj_rejected:<24} │")
        print(f"  │  Avg Confidence        │  {avg_obj_conf:.4f}                   │")
        print(f"  │  Max Confidence        │  {max_obj_conf:.4f}                   │")
        print(f"  │  Min Confidence        │  {min_obj_conf:.4f}                   │")
        print(f"  └────────────────────────┴──────────────────────────┘")

        print(f"\n  ┌───────────────────────────────────────────────────┐")
        print(f"  │  MTCNN FACE DETECTOR                              │")
        print(f"  ├────────────────────────┬──────────────────────────┤")
        print(f"  │  Inference Frames      │  {face_inf:<24} │")
        print(f"  │  Face Detected         │  {face_detected:<24} │")
        print(f"  │  Face Absent           │  {face_absent:<24} │")
        print(f"  │  Violation Frames (>5s)│  {face_violations:<24} │")
        print(f"  │  Avg MTCNN Confidence  │  {avg_face_conf:.4f}                   │")
        print(f"  └────────────────────────┴──────────────────────────┘")

        # Per-detector violation triggers
        if self.per_detector_triggers:
            print(f"\n  ┌───────────────────────────────────────────────────┐")
            print(f"  │  PER-DETECTOR VIOLATION TRIGGERS                  │")
            print(f"  ├────────────────────────┬──────────────────────────┤")
            for name, count in sorted(self.per_detector_triggers.items()):
                pct = (count / self.total_frames * 100) if self.total_frames > 0 else 0
                print(f"  │  {name:<22s} │  {count:>5} frames ({pct:>5.1f}%)   │")
            print(f"  └────────────────────────┴──────────────────────────┘")

        print("\n" + "=" * 66)
        print("  Ground Truth: Press 'v' during session to mark violations")
        print("  TP = detected + confirmed  │  FP = detected + not confirmed")
        print("  TN = clean + confirmed     │  FN = missed + was violation")
        print("  MCC range: -1 (worst) → 0 (random) → +1 (perfect)")
        print("=" * 66 + "\n")


# ══════════════════════════════════════════════════════════════════════════
#  DISPLAY & VIOLATION HANDLING
# ══════════════════════════════════════════════════════════════════════════

def display_detection_results(frame, results):
    y_offset = 30
    line_height = 30
    
    # Status indicators
    status_items = [
        f"Face: {'Present' if results['face_present'] else 'Absent'}",
        f"Gaze: {results['gaze_direction']}",
        f"Eyes: {'Open' if results['eye_ratio'] > 0.25 else 'Closed'}",
        f"Mouth: {'Moving' if results['mouth_moving'] else 'Still'}"
    ]
    
    # Alert indicators
    alert_items = []
    if results['multiple_faces']:
        alert_items.append("Multiple Faces Detected!")
    if results['objects_detected']:
        alert_items.append("Suspicious Object Detected!")
        
    # Massive Cheating Pop-ups
    popups = []
    if not results.get('face_present', True):
        popups.append("ALERT: FACE NOT DETECTED")
    if results.get('eye_alarming'):
        popups.append("SUSPICIOUS: EXCESSIVE EYE MOVEMENT")
    if results.get('mouth_alarming'):
        popups.append("CHEATING: WHISPERING / TALKING")
    if results.get('objects_detected') and results.get('detected_object_label'):
        labels = results['detected_object_label'].upper()
        if labels == "CELL PHONE":
            popups.append("UNAUTHORIZED CELL PHONE DETECTED")
        elif labels == "UNIDENTIFIED OBJECT":
            popups.append("UNAUTHORIZED OBJECT DETECTED")
        else:
            popups.append(f"UNAUTHORIZED OBJECT DETECTED: {labels}")
    if results.get('hand_violation') and results.get('hand_violation_msg'):
        popups.append(results['hand_violation_msg'].upper())
        
    if popups:
        # Draw a semi-transparent red box across the top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], len(popups)*45 + 20), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        y_pop = 40
        for text in popups:
            cv2.putText(frame, text, (frame.shape[1]//2 - 300, y_pop), 
                       cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
            y_pop += 40

    # Display status
    for item in status_items:
        cv2.putText(frame, item, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += line_height
    
    # Display alerts
    for item in alert_items:
        cv2.putText(frame, item, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += line_height
    
    # Timestamp
    cv2.putText(frame, results['timestamp'], 
               (frame.shape[1] - 250, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


last_violation_times = {}

def handle_violation(violation_type, frame, results, alert_system, violation_capturer, violation_logger, custom_message=None):
    """Unified handler for all violation types"""
    current_time = time.time()
    cooldown = alert_system.config.get('logging', {}).get('alert_cooldown', 5)
    last_time = last_violation_times.get(violation_type, 0)
    
    if current_time - last_time < cooldown:
        return
        
    last_violation_times[violation_type] = current_time

    alert_system.speak_alert(violation_type, custom_message=custom_message)
    
    # Capture and log violation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    violation_image = violation_capturer.capture_violation(frame, violation_type, timestamp)
    
    metadata = {'duration': 'Detected', 'frame': results}
    if violation_image:
        metadata['image_path'] = violation_image['image_path']
        
    violation_logger.log_violation(
        violation_type,
        timestamp,
        metadata
    )


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    config = load_config()
    alert_logger = AlertLogger(config)
    alert_system = AlertSystem(config)
    violation_capturer = ViolationCapturer(config)
    violation_logger = ViolationLogger(config)
    report_generator = ReportGenerator(config)
    metrics = MetricsTracker()

    # Student info could eventually come from a login or command line
    student_info = {
        'id': 'STUDENT_001',
        'name': 'John Doe',
        'exam': 'Final Examination',
        'course': 'Computer Science 101'
    }

    # Video/Screen recording has been disabled per user request
    # video_recorder = VideoRecorder(config)
    # screen_recorder = ScreenRecorder(config)
    
    # Initialize hardware monitor
    hardware_monitor = HardwareMonitor(config)
    hardware_monitor.set_alert_logger(alert_logger)
    hardware_monitor.start()

    # Initialize audio monitor
    audio_monitor = AudioMonitor(config)
    audio_monitor.alert_system = alert_system
    audio_monitor.alert_logger = alert_logger

    audio_started = False
    if config['detection']['audio_monitoring']['enabled']:
        audio_started = audio_monitor.start()
        if not audio_started:
            print("Warning: Audio monitoring failed to start. Continuing with visual detection only.")

    cap = None
    try:
        # Hardware Check: Webcam
        cap = cv2.VideoCapture(config['video']['source'])
        if not cap.isOpened():
            print(f"Error: Could not open video source {config['video']['source']}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['video']['resolution'][0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['video']['resolution'][1])
        
        # Verify we can actually read a frame
        ret, test_frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            return

        # Start recordings has been disabled
        # if config['screen']['recording']:
        #     screen_recorder.start_recording()
        # video_recorder.start_recording()

        # Initialize detectors safely
        detector_classes = [
            ObjectDetector,
            FaceDetector,
            EyeTracker,
            MouthMonitor,
            MultiFaceDetector
        ]
        
        # Add HandMonitor if available
        try:
            from detection.hand_detection import HandMonitor
            detector_classes.append(HandMonitor)
        except ImportError:
            pass
            
        detectors = []
        for cls in detector_classes:
            try:
                det = cls(config)
                if hasattr(det, 'set_alert_logger'):
                    det.set_alert_logger(alert_logger)
                detectors.append(det)
            except Exception as e:
                print(f"Warning: Failed to initialize {cls.__name__}: {e}")
        
        if not detectors:
            print("Error: No detectors could be initialized. Exiting.")
            return

        print("System started successfully.")
        print("  Press 'q' to quit")
        print("  Press 'v' to toggle ground-truth violation marker")
        print("  (Mark ground truth to get meaningful metrics)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            results = {
                'face_present': False,
                'gaze_direction': 'Center',
                'eye_ratio': 0.3,
                'mouth_moving': False,
                'multiple_faces': False,
                'objects_detected': False,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            person_present = False
            # Perform detections safely
            for det in detectors:
                if isinstance(det, ObjectDetector):
                    results['objects_detected'], person_present = det.detect_objects(frame)
                    _, results['detected_object_label'] = det.is_alarming()
                elif isinstance(det, FaceDetector):
                    results['face_present'] = det.detect_face(frame, fallback_person_present=person_present)
                elif isinstance(det, EyeTracker):
                    face_detector_instance = next((d for d in detectors if isinstance(d, FaceDetector)), None)
                    lms = face_detector_instance.last_landmarks if face_detector_instance else None
                    results['gaze_direction'], results['eye_ratio'] = det.track_eyes(frame, fallback_landmarks=lms)
                    results['eye_alarming'] = det.is_alarming()
                elif isinstance(det, MouthMonitor):
                    face_detector_instance = next((d for d in detectors if isinstance(d, FaceDetector)), None)
                    lms = face_detector_instance.last_landmarks if face_detector_instance else None
                    results['mouth_moving'] = det.monitor_mouth(frame, fallback_landmarks=lms)
                    results['mouth_alarming'] = det.is_alarming()
                elif isinstance(det, MultiFaceDetector):
                    results['multiple_faces'] = det.detect_multiple_faces(frame)
                elif type(det).__name__ == "HandMonitor":
                    hand_alert_triggered, hand_alert_msg = det.monitor_hands(frame)
                    if hand_alert_triggered:
                        results['hand_violation'] = True
                        results['hand_violation_msg'] = hand_alert_msg

            # ── Determine if this frame has a violation ────────────────
            frame_violation = False
            active_detectors = []

            face_detector_instance = next((d for d in detectors if isinstance(d, FaceDetector)), None)

            if face_detector_instance and face_detector_instance.is_violation():
                frame_violation = True
                active_detectors.append("FACE_DISAPPEARED")
                handle_violation("FACE_DISAPPEARED", frame, results, alert_system, violation_capturer, violation_logger)

            if results.get('multiple_faces'):
                frame_violation = True
                active_detectors.append("MULTIPLE_FACES")
                if "FACE_DISAPPEARED" not in active_detectors:
                    handle_violation("MULTIPLE_FACES", frame, results, alert_system, violation_capturer, violation_logger)

            if results.get('objects_detected'):
                frame_violation = True
                labels = results.get('detected_object_label', '')
                active_detectors.append("OBJECT_DETECTED")
                if labels == "cell phone":
                    msg = "Unauthorized cell phone detected."
                elif labels == "unidentified object":
                    msg = "Unauthorized object detected."
                else:
                    msg = f"Unauthorized object detected: {labels}."
                handle_violation("OBJECT_DETECTED", frame, results, alert_system, violation_capturer, violation_logger, custom_message=msg)

            if results.get('mouth_alarming'):
                frame_violation = True
                active_detectors.append("MOUTH_MOVING")
                handle_violation("MOUTH_MOVING", frame, results, alert_system, violation_capturer, violation_logger)

            if results.get('hand_violation'):
                frame_violation = True
                active_detectors.append("HAND_VIOLATION")
                handle_violation("HAND_VIOLATION", frame, results, alert_system, violation_capturer, violation_logger, custom_message=results.get('hand_violation_msg'))

            # ── Record frame for metrics ───────────────────────────────
            metrics.record_frame(frame_violation, active_detectors if frame_violation else None)

            # Show GT indicator on frame
            if metrics.ground_truth_active:
                cv2.rectangle(frame, (frame.shape[1]-220, frame.shape[0]-45), 
                             (frame.shape[1]-5, frame.shape[0]-5), (0, 0, 200), -1)
                cv2.putText(frame, "GT: VIOLATION", (frame.shape[1]-215, frame.shape[0]-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "GT: CLEAN", (frame.shape[1]-150, frame.shape[0]-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

            # Display
            display_detection_results(frame, results)
            
            # Show preview and handle key input
            cv2.imshow('Exam Proctoring', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('v'):
                metrics.toggle_ground_truth()
                
    except KeyboardInterrupt:
        print("\nSession stopped by user (Ctrl+C).")
    except Exception as e:
        print(f"An unexpected error occurred during execution: {e}")
    finally:
        print("Cleaning up resources...")
        
        # Stop everything
        try:
            hardware_monitor.stop()
        except: pass
        
        if audio_started:
            audio_monitor.stop()
            
        # Screen/Video recording stopping disabled
        # if config['screen']['recording']:
        #     try:
        #         screen_recorder.stop_recording()
        #     except: pass
        #     
        # try:
        #     video_recorder.stop_recording()
        # except: pass

        if cap and cap.isOpened():
            cap.release()
            
        cv2.destroyAllWindows()

        # ── Print Performance Metrics (from actual model data) ─────────
        metrics.print_report(detectors)

        # Generate report at the very end
        try:
            violations = violation_logger.get_violations()
            report_path = report_generator.generate_report(student_info, violations)
            if report_path:
                print(f"Session complete. Report generated at: {report_path}")
        except Exception as e:
            print(f"Error during report generation: {e}")

if __name__ == '__main__':
    main()