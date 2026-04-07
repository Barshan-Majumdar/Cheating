import pyaudio
import numpy as np
import threading
from collections import deque
try:
    import whisper
    WHISPER_AVAILABLE = True
except (ImportError, Exception):
    WHISPER_AVAILABLE = False
import time

class AudioMonitor:
    def __init__(self, config):
        self.config = config['detection']['audio_monitoring']
        self.sample_rate = self.config['sample_rate']
        self.chunk_size = 512  # 32ms chunks for low latency
        self.energy_threshold = self.config['energy_threshold']
        self.zcr_threshold = self.config['zcr_threshold']
        self.running = False
        self.audio_buffer = deque(maxlen=15)  # 480ms buffer
        self.alert_system = None
        self.alert_logger = None
        
        if self.config['whisper_enabled'] and WHISPER_AVAILABLE:
            try:
                self.whisper_model = whisper.load_model(self.config['whisper_model'])
            except Exception as e:
                print(f"Failed to load Whisper model: {e}")
                self.config['whisper_enabled'] = False
        else:
            if self.config['whisper_enabled']:
                print("Whisper is enabled but the library is not compatible with this Python version. Speech transcription will be disabled.")
                self.config['whisper_enabled'] = False
        
    def start(self):
        """Start audio monitoring thread with hardware check"""
        try:
            p = pyaudio.PyAudio()
            # Try to open a temporary stream to check hardware
            test_stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            test_stream.close()
            p.terminate()
        except Exception as e:
            if self.alert_logger:
                self.alert_logger.log_alert("AUDIO_HARDWARE_ERROR", f"Could not access microphone: {e}")
            print(f"Audio Hardware Error: {e}")
            return False

        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return True
        
    def stop(self):
        """Stop audio monitoring safely"""
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=2)
            
    def _run(self):
        """Main audio processing loop with robustness"""
        p = pyaudio.PyAudio()
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
        except Exception as e:
            print(f"Critical error opening audio stream: {e}")
            p.terminate()
            return
        
        try:
            while self.running:
                try:
                    # Using small chunks to keep the loop responsive to self.running
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    audio = np.frombuffer(data, dtype=np.int16)
                    self.audio_buffer.append(audio)
                    
                    if self._is_voice(audio):
                        self._handle_voice_detection()
                except Exception as e:
                    print(f"Error reading audio stream: {e}")
                    time.sleep(0.1)
                    
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except:
                pass
            p.terminate()
    
    def _is_voice(self, audio):
        """Ultra-fast voice detection"""
        audio_norm = audio / 32768.0
        
        # 1. Energy detection
        energy = np.mean(audio_norm**2)
        if energy < self.energy_threshold:
            return False
            
        # 2. Zero-crossing rate
        zcr = np.mean(np.abs(np.diff(np.sign(audio_norm))))
        if zcr > self.zcr_threshold:
            return False
            
        return True
    
    def _handle_voice_detection(self):
        """Process detected voice"""
        if self.alert_system:
            self.alert_system.speak_alert("VOICE_DETECTED")
            
        if self.alert_logger:
            self.alert_logger.log_alert("VOICE_DETECTED", "Voice activity detected")
            
        if self.config['whisper_enabled'] and WHISPER_AVAILABLE:
            self._process_with_whisper()
    
    def _process_with_whisper(self):
        """Optional Whisper processing"""
        try:
            audio = np.concatenate(self.audio_buffer)
            result = self.whisper_model.transcribe(
                audio.astype(np.float32) / 32768.0,
                fp16=False,
                language='en'
            )
            
            text = result['text'].strip().lower()
            if any(word in text for word in ['help', 'answer', 'whisper']):
                if self.alert_system:
                    self.alert_system.speak_alert("SPEECH_VIOLATION")
                    
        except Exception as e:
            if self.alert_logger:
                self.alert_logger.log_alert("WHISPER_ERROR", str(e))