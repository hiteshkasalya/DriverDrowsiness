import os
import sys
import time
import threading

# Suppress pygame prompt
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

if sys.platform == "win32":
    try:
        import winsound
        WINSOUND_AVAILABLE = True
    except ImportError:
        WINSOUND_AVAILABLE = False
else:
    WINSOUND_AVAILABLE = False


class AlertSystem:
    """
    Robust non-blocking audio alert system for Driver Drowsiness Detection.
    Features:
    - Auto-locates alarm sound across relative paths
    - Plays audio asynchronously without blocking the video pipeline
    - Fallback to winsound (Windows) or terminal alert if pygame fails
    - Multi-level alerting: Warning vs Critical Sleep Alarm
    - Configurable closure duration threshold and tolerance buffer
    """

    def __init__(self, alarm_path=None, eye_closed_sec=1.5, tolerance_sec=0.3):
        self.eye_closed_sec = eye_closed_sec
        self.tolerance_sec = tolerance_sec

        self.closed_start = None
        self.last_closed_time = None
        self.alarm_playing = False
        self.sound = None
        self.channel = None
        self.sound_enabled = True

        self._stop_beep_thread = False
        self._beep_thread = None

        self.resolved_path = self._resolve_alarm_path(alarm_path)
        self._init_audio()

    def _resolve_alarm_path(self, user_path):
        candidates = []
        if user_path:
            candidates.append(user_path)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(base_dir, ".."))
        parent_root = os.path.abspath(os.path.join(project_root, ".."))

        # Candidate relative paths
        search_dirs = [project_root, parent_root, os.getcwd()]
        for d in search_dirs:
            candidates.append(os.path.join(d, "assets", "alarm.mp3"))
            candidates.append(os.path.join(d, "models", "alarm.mp3"))
            candidates.append(os.path.join(d, "assests", "alarm.mp3"))
            candidates.append(os.path.join(d, "minipro", "assets", "alarm.mp3"))
            candidates.append(os.path.join(d, "minipro", "assests", "alarm.mp3"))

        for path in candidates:
            if path and os.path.isfile(path):
                return os.path.abspath(path)

        return None

    def _init_audio(self):
        if not PYGAME_AVAILABLE:
            return

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            if self.resolved_path and os.path.exists(self.resolved_path):
                self.sound = pygame.mixer.Sound(self.resolved_path)
                self.sound.set_volume(1.0)
        except Exception as e:
            # Pygame audio device could be busy or not present
            self.sound = None

    def update(self, eye_status, is_yawning=False, is_nodding=False):
        """
        Updates drowsiness status:
        - eye_status: "Eyes Closed", "Eyes Open", or "Face not detected"
        - is_yawning: bool
        - is_nodding: bool
        Returns dict with current state metrics.
        """
        now = time.time()
        is_closed = (eye_status == "Eyes Closed")

        if is_closed:
            self.last_closed_time = now
            if self.closed_start is None:
                self.closed_start = now

            closed_duration = now - self.closed_start
            if closed_duration >= self.eye_closed_sec:
                self.start_alarm()
                return {
                    "alert_level": "CRITICAL",
                    "closed_duration": closed_duration,
                    "alarm_playing": self.alarm_playing,
                    "message": f"CRITICAL: SLEEPING DETECTED! ({closed_duration:.1f}s)"
                }
            elif closed_duration >= (self.eye_closed_sec * 0.5):
                return {
                    "alert_level": "WARNING",
                    "closed_duration": closed_duration,
                    "alarm_playing": self.alarm_playing,
                    "message": f"WARNING: Eyes closing ({closed_duration:.1f}s)"
                }
            else:
                return {
                    "alert_level": "NORMAL",
                    "closed_duration": closed_duration,
                    "alarm_playing": self.alarm_playing,
                    "message": "Eyes Closed (Blink)"
                }
        else:
            # If eyes are open or face temporarily lost
            if self.last_closed_time is not None:
                time_since_closed = now - self.last_closed_time
                if time_since_closed < self.tolerance_sec:
                    # Within tolerance window: do not reset closed_start
                    closed_duration = (self.last_closed_time - self.closed_start) if self.closed_start else 0.0
                    return {
                        "alert_level": "WARNING" if closed_duration >= (self.eye_closed_sec * 0.5) else "NORMAL",
                        "closed_duration": closed_duration,
                        "alarm_playing": self.alarm_playing,
                        "message": "Tracking..."
                    }

            # Beyond tolerance: reset closed state
            self.closed_start = None
            self.last_closed_time = None
            if self.alarm_playing:
                self.stop_alarm()

            if is_yawning:
                return {
                    "alert_level": "WARNING",
                    "closed_duration": 0.0,
                    "alarm_playing": False,
                    "message": "WARNING: YAWNING DETECTED!"
                }
            if is_nodding:
                return {
                    "alert_level": "WARNING",
                    "closed_duration": 0.0,
                    "alarm_playing": False,
                    "message": "WARNING: HEAD NODDING DETECTED!"
                }

            return {
                "alert_level": "NORMAL",
                "closed_duration": 0.0,
                "alarm_playing": False,
                "message": eye_status
            }

    def start_alarm(self):
        if not self.alarm_playing:
            self.alarm_playing = True

            if not self.sound_enabled:
                return

            if self.sound:
                try:
                    self.sound.play(loops=-1)
                    return
                except Exception:
                    pass

            # Fallback to background winsound beep
            if WINSOUND_AVAILABLE and (self._beep_thread is None or not self._beep_thread.is_alive()):
                self._stop_beep_thread = False
                self._beep_thread = threading.Thread(target=self._beep_loop, daemon=True)
                self._beep_thread.start()

    def _beep_loop(self):
        while not self._stop_beep_thread and self.alarm_playing:
            if WINSOUND_AVAILABLE:
                try:
                    winsound.Beep(1800, 300)
                except Exception:
                    break
            time.sleep(0.05)

    def stop_alarm(self):
        self.alarm_playing = False
        self._stop_beep_thread = True
        if self.sound:
            try:
                self.sound.stop()
            except Exception:
                pass

    def stop(self):
        self.stop_alarm()
        self.closed_start = None
        self.last_closed_time = None
