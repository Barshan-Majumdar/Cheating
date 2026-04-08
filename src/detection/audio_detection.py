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
        self.chunk_size = 512
        self.energy_threshold = self.config['energy_threshold']
        self.zcr_threshold = self.config['zcr_threshold']
        self.running = False
        self.audio_buffer = deque(maxlen=15)
        self.alert_system = None
        self.alert_logger = None
        self.consecutive_voice_frames = 0
        self.voice_frame_threshold = 8
        
        # Audio backend: try sounddevice first, then pyaudio
        self.backend = None
        self.stream_channels = 1
        self.stream_device = None
        
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
        
    def _try_sounddevice(self):
        """Try to find a working sounddevice configuration"""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            
            # Try each input device with various configs
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    for rate in [int(dev['default_samplerate']), 44100, 48000, 16000]:
                        for ch in [1, min(2, dev['max_input_channels'])]:
                            try:
                                test = sd.rec(int(0.1 * rate), samplerate=rate,
                                            channels=ch, dtype='int16', device=i)
                                sd.wait()
                                self.sample_rate = rate
                                self.stream_channels = ch
                                self.stream_device = i
                                self.backend = 'sounddevice'
                                return True
                            except Exception:
                                continue
        except ImportError:
            pass
        return False
    
    def _try_pyaudio(self):
        """Try to find a working PyAudio configuration"""
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    for rate in [int(info['defaultSampleRate']), 44100, 48000, 16000]:
                        for ch in [1, min(2, info['maxInputChannels'])]:
                            try:
                                s = p.open(format=pyaudio.paInt16, channels=ch,
                                          rate=rate, input=True,
                                          input_device_index=i,
                                          frames_per_buffer=self.chunk_size)
                                s.close()
                                self.sample_rate = rate
                                self.stream_channels = ch
                                self.stream_device = i
                                self.backend = 'pyaudio'
                                p.terminate()
                                return True
                            except Exception:
                                continue
            p.terminate()
        except ImportError:
            pass
        return False
        
    def start(self):
        """Start audio monitoring thread"""
        if self._try_pyaudio() or self._try_sounddevice():
            print(f"Audio monitoring started ({self.backend}, device {self.stream_device}, {self.sample_rate}Hz, {self.stream_channels}ch)")
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            return True
        else:
            print("Warning: No working audio input found. Audio monitoring disabled.")
            return False
        
    def stop(self):
        """Stop audio monitoring safely"""
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=2)
            
    def _run(self):
        """Main audio processing loop"""
        if self.backend == 'pyaudio':
            self._run_pyaudio()
        elif self.backend == 'sounddevice':
            self._run_sounddevice()
    
    def _run_pyaudio(self):
        """PyAudio-based audio loop"""
        import pyaudio
        p = pyaudio.PyAudio()
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=self.stream_channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.stream_device,
                frames_per_buffer=self.chunk_size
            )
        except Exception as e:
            print(f"Critical error opening audio stream: {e}")
            p.terminate()
            return
        
        try:
            while self.running:
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    audio = np.frombuffer(data, dtype=np.int16)
                    if self.stream_channels == 2:
                        audio = audio[::2]
                    self._process_audio(audio)
                except Exception as e:
                    time.sleep(0.1)
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except:
                pass
            p.terminate()
    
    def _run_sounddevice(self):
        """sounddevice-based audio loop"""
        import sounddevice as sd
        
        try:
            while self.running:
                try:
                    audio = sd.rec(int(self.chunk_size), samplerate=self.sample_rate,
                                  channels=self.stream_channels, dtype='int16',
                                  device=self.stream_device)
                    sd.wait()
                    audio = audio.flatten()
                    if self.stream_channels == 2:
                        audio = audio[::2]
                    self._process_audio(audio)
                except Exception:
                    time.sleep(0.1)
        except Exception:
            pass
    
    def _process_audio(self, audio):
        """Process an audio chunk for voice detection"""
        self.audio_buffer.append(audio)
        
        if self._is_voice(audio):
            self.consecutive_voice_frames += 1
            if self.consecutive_voice_frames >= self.voice_frame_threshold:
                self._handle_voice_detection()
                self.consecutive_voice_frames = 0
        else:
            self.consecutive_voice_frames = max(0, self.consecutive_voice_frames - 2)
    
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