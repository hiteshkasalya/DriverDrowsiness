import os
import sys
import time
import cv2

# Set path to current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.drowsiness_detector import ComprehensiveDrowsinessDetector


def print_banner():
    print("=" * 65)
    print("      DRIVER DROWSINESS DETECTION SYSTEM (REAL-TIME)")
    print("=" * 65)
    print("  Controls:")
    print("    [Q] / [ESC]  : Quit application")
    print("    [M]          : Switch Detection Mode (Hybrid / FaceMesh / CNN / Haar)")
    print("    [A]          : Toggle Sound Alarm (ON/OFF)")
    print("    [+] / [-]    : Adjust Eye Closure Sensitivity (EAR threshold)")
    print("    [R]          : Reset Statistics (Blinks, Yawns, Fatigue)")
    print("=" * 65)


def run_desktop_app():
    print_banner()

    detector = ComprehensiveDrowsinessDetector(
        ear_threshold=0.22,
        mar_threshold=0.60,
        eye_closed_sec=1.5,
        model_type="hybrid",
        enable_sound=True
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam. Trying camera index 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("[FATAL] No accessible webcam found! Please connect a camera.")
            detector.close()
            return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = "Driver Drowsiness Detection System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    modes = ["hybrid", "facemesh", "cnn", "haar"]
    mode_idx = 0

    print("\n[INFO] Starting live video stream. Keep your face visible to the camera...\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Empty frame received from webcam. Retrying...")
                time.sleep(0.05)
                continue

            # Mirror frame for natural driver view
            frame = cv2.flip(frame, 1)

            # Process frame through detection engine
            annotated_frame, metrics = detector.process_frame(frame)

            # Render window
            cv2.imshow(window_name, annotated_frame)

            # Handle user keys
            key = cv2.waitKey(1) & 0xFF

            if key in [ord('q'), ord('Q'), 27]:  # Quit
                print("\n[INFO] Exiting...")
                break

            elif key in [ord('m'), ord('M')]:  # Switch mode
                mode_idx = (mode_idx + 1) % len(modes)
                new_mode = modes[mode_idx]
                detector.model_type = new_mode
                print(f"[MODE] Switched to: {new_mode.upper()}")

            elif key in [ord('a'), ord('A')]:  # Toggle audio
                detector.alert.sound_enabled = not detector.alert.sound_enabled
                status = "ON" if detector.alert.sound_enabled else "MUTED"
                print(f"[AUDIO] Sound alarm is now {status}")
                if not detector.alert.sound_enabled:
                    detector.alert.stop_alarm()

            elif key in [ord('+'), ord('=')]:  # Increase EAR threshold
                detector.ear_threshold = round(min(0.35, detector.ear_threshold + 0.01), 2)
                print(f"[SETTING] EAR Threshold increased to: {detector.ear_threshold}")

            elif key in [ord('-'), ord('_')]:  # Decrease EAR threshold
                detector.ear_threshold = round(max(0.12, detector.ear_threshold - 0.01), 2)
                print(f"[SETTING] EAR Threshold decreased to: {detector.ear_threshold}")

            elif key in [ord('r'), ord('R')]:  # Reset stats
                detector.blink_count = 0
                detector.yawn_count = 0
                detector.nod_count = 0
                detector.drowsiness_score = 0
                detector.alert.stop_alarm()
                print("[STATS] Counters reset to 0.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print("[INFO] Cleanup complete. System stopped.")


if __name__ == "__main__":
    run_desktop_app()
