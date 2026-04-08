import cv2
import yaml
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
    if results.get('eye_alarming'):
        popups.append("SUSPICIOUS: EXCESSIVE EYE MOVEMENT")
    if results.get('mouth_alarming'):
        popups.append("CHEATING: WHISPERING / TALKING")
    if results.get('objects_detected') and results.get('detected_object_label'):
        popups.append(f"UNAUTHORIZED OBJECT: {results['detected_object_label'].upper()}")
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

def handle_violation(violation_type, frame, results, alert_system, violation_capturer, violation_logger, custom_message=None):
    """Unified handler for all violation types"""
    alert_system.speak_alert(violation_type, custom_message=custom_message)
    
    # Capture and log violation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    violation_image = violation_capturer.capture_violation(frame, violation_type, timestamp)
    violation_logger.log_violation(
        violation_type,
        timestamp,
        {'duration': 'Detected', 'frame': results}
    )

def main():
    config = load_config()
    alert_logger = AlertLogger(config)
    alert_system = AlertSystem(config)
    violation_capturer = ViolationCapturer(config)
    violation_logger = ViolationLogger(config)
    report_generator = ReportGenerator(config)

    # Student info could eventually come from a login or command line
    student_info = {
        'id': 'STUDENT_001',
        'name': 'John Doe',
        'exam': 'Final Examination',
        'course': 'Computer Science 101'
    }

    # Initialize recorders
    video_recorder = VideoRecorder(config)
    screen_recorder = ScreenRecorder(config)
    
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

        # Start recordings
        if config['screen']['recording']:
            screen_recorder.start_recording()
        video_recorder.start_recording()

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
            from src.detection.hand_detection import HandMonitor
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

        print("System started successfully. Press 'q' to quit.")
        
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

            # Violation Checks
            face_detector_instance = next((d for d in detectors if isinstance(d, FaceDetector)), None)
            
            if face_detector_instance and face_detector_instance.is_violation():
                handle_violation("FACE_DISAPPEARED", frame, results, alert_system, violation_capturer, violation_logger)
            elif results.get('multiple_faces'):
                handle_violation("MULTIPLE_FACES", frame, results, alert_system, violation_capturer, violation_logger)
            elif results.get('objects_detected'):
                handle_violation("OBJECT_DETECTED", frame, results, alert_system, violation_capturer, violation_logger, custom_message=f"Unauthorized object {results.get('detected_object_label', '')} detected")
            elif results.get('mouth_moving'):
                handle_violation("MOUTH_MOVING", frame, results, alert_system, violation_capturer, violation_logger)
            elif results.get('hand_violation'):
                handle_violation("HAND_VIOLATION", frame, results, alert_system, violation_capturer, violation_logger, custom_message=results.get('hand_violation_msg'))

            # Display and record
            display_detection_results(frame, results)
            video_recorder.record_frame(frame)
            
            # Show preview
            cv2.imshow('Exam Proctoring', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
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
            
        if config['screen']['recording']:
            try:
                screen_recorder.stop_recording()
            except: pass
            
        try:
            video_recorder.stop_recording()
        except: pass
        
        if cap and cap.isOpened():
            cap.release()
            
        cv2.destroyAllWindows()

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