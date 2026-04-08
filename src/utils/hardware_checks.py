import psutil
import time
import win32api
import win32con
import threading
import pythoncom
try:
    from pygrabber.dshow_graph import FilterGraph  # type: ignore
    PYGRABBER_AVAILABLE = True
except Exception as e:
    PYGRABBER_AVAILABLE = False
    print(f"Warning: pygrabber not available. Virtual camera checks disabled. Error: {e}")

class HardwareMonitor:
    def __init__(self, config=None):
        self.config = config or {}
        self.alert_logger = None
        self.running = False
        
        # Blacklists
        self.forbidden_cameras = ["OBS", "Virtual", "ManyCam", "Snap", "XSplit", "EpocCam", "DroidCam"]
        self.forbidden_processes = ["TeamViewer", "Discord", "AnyDesk", "Skype", "Zoom", "Slack", "obs64.exe"]
        
    def set_alert_logger(self, alert_logger):
        self.alert_logger = alert_logger
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_monitoring_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=2)
            
    def _run_monitoring_loop(self):
        # COM initialization must happen on the thread that calls pygrabber/COM objects
        pythoncom.CoInitialize()
        try:
            while self.running:
                self._check_virtual_cameras()
                self._check_multiple_displays()
                self._check_forbidden_processes()
                
                # Polling interval of 10 seconds for ultra-low load
                for _ in range(10):
                    if not self.running:
                        break
                    time.sleep(1)
        finally:
            pythoncom.CoUninitialize()

    def _check_virtual_cameras(self):
        if not PYGRABBER_AVAILABLE:
            return
            
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()
            
            for device in devices:
                dev_name = device.lower()
                for forbidden in self.forbidden_cameras:
                    if forbidden.lower() in dev_name:
                        if self.alert_logger:
                            self.alert_logger.log_alert(
                                "VIRTUAL_CAMERA_DETECTED",
                                f"Forbidden camera detected: {device}"
                            )
        except Exception as e:
            print(f"Failed to check virtual cameras: {e}")

    def _check_multiple_displays(self):
        try:
            monitors = win32api.EnumDisplayMonitors()
            if len(monitors) > 1:
                if self.alert_logger:
                    self.alert_logger.log_alert(
                        "MULTIPLE_DISPLAYS_DETECTED",
                        f"Detected {len(monitors)} active displays"
                    )
        except Exception as e:
            print(f"Failed to enumerate displays: {e}")

    def _check_forbidden_processes(self):
        try:
            # Iterating through running processes is fast, but doing it sparsely saves CPU
            for proc in psutil.process_iter(['name']):
                proc_name = proc.info['name']
                if not proc_name:
                    continue
                
                proc_name_lower = proc_name.lower()
                for forbidden in self.forbidden_processes:
                    if forbidden.lower() in proc_name_lower:
                        if self.alert_logger:
                            self.alert_logger.log_alert(
                                "FORBIDDEN_PROCESS_DETECTED",
                                f"Active forbidden software: {proc_name}"
                            )
                        # We break instead of logging every single discord thread
                        break
        except Exception as e:
            pass # psutil might throw AccessDenied on some protected processes

if __name__ == "__main__":
    monitor = HardwareMonitor()
    monitor._check_virtual_cameras()
    monitor._check_multiple_displays()
    monitor._check_forbidden_processes()
    print("Done hardware checks.")
