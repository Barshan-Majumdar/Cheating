import os
import tempfile
from gtts import gTTS
import pygame
import threading
import time

class AlertSystem:
    def __init__(self, config):
        pygame.mixer.init()
        self.config = config
        self.alert_cooldown = config['logging']['alert_cooldown']
        self.last_alert_time = {}
        
        # Alert messages database
        self.alerts = {
            "FACE_DISAPPEARED": "Please look at the screen",
            "FACE_REAPPEARED": "Thank you for looking at the screen",
            "MULTIPLE_FACES": "We detected multiple people",
            "OBJECT_DETECTED": "Unauthorized object detected",
            "GAZE_AWAY": "Please focus on your screen",
            "MOUTH_MOVING": "Cheating detected. Whispering or talking is not allowed.",
            "SPEECH_VIOLATION": "Cheating detected. Speaking during the exam is not allowed.",
            "VOICE_DETECTED": "Cheating detected. Voice activity discovered. Please remain silent.",
            "HAND_VIOLATION": "Suspicious hand movements detected",
        }
        
    def _can_alert(self, alert_type):
        """Check if enough time has passed since last alert"""
        current_time = time.time()
        last_time = self.last_alert_time.get(alert_type, 0)
        return (current_time - last_time) >= self.alert_cooldown
        
    def speak_alert(self, alert_type, custom_message=None):
        """Convert text to speech and play it"""
        voice_enabled = self.config.get('logging', {}).get('alert_system', {}).get('voice_alerts', True)
        if not voice_enabled or not self._can_alert(alert_type):
            return
            
        self.last_alert_time[alert_type] = time.time()
        
        def _play_audio():
            try:
                # Dynamic audio synthesis
                message = custom_message if custom_message else self.alerts.get(alert_type)
                if message:
                    # Generate speech
                    tts = gTTS(text=message, lang='en')
                    
                    # Save temporary audio file
                    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as fp:
                        temp_path = fp.name
                        tts.save(temp_path)
                    
                    # Play audio
                    pygame.mixer.music.load(temp_path)
                    pygame.mixer.music.play()
                    
                    # Wait until playback finishes
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    
                    # Release the file handle on Windows
                    pygame.mixer.music.unload()
                    
                    # Cleanup with retries for Windows file locking
                    max_retries = 5
                    for i in range(max_retries):
                        try:
                            if os.path.exists(temp_path):
                                os.unlink(temp_path)
                            break
                        except PermissionError:
                            if i < max_retries - 1:
                                time.sleep(0.5)
                            else:
                                print(f"Warning: Could not cleanup temporary audio file {temp_path}")
            except Exception as e:
                print(f"Audio alert failed: {str(e)}")
        
        # Run in separate thread to avoid blocking
        threading.Thread(target=_play_audio, daemon=True).start()
