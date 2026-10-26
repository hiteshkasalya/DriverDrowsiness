import os
import sys
import time
import math
import numpy as np
import cv2

# Suppress TensorFlow & MediaPipe logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False

try:
    import keras
    KERAS_AVAILABLE = True
except Exception:
    KERAS_AVAILABLE = False

from modules.eye_detection import calculate_ear, load_eye_cascade, detect_eyes, predict_eye_status
from modules.yawning_detection import calculate_mar_from_landmarks, load_face_cascade, detect_mouth
from modules.alert_system import AlertSystem


class ComprehensiveDrowsinessDetector:
    """
    High-Performance Multi-Modal Driver Drowsiness Detector.
    Supports:
    - MediaPipe FaceLandmarker with EAR & Blendshapes
    - Mouth Aspect Ratio (MAR) for precise Yawn Detection
    - 3D Head Pose & Nodding (Pitch/Roll) estimation
    - Hybrid CNN Verification (eye_model.keras / drowsiness_model.keras)
    - Fallback Haar Cascade detector
    - Real-time HUD graphics and stats
    """

    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

    def __init__(
        self,
        ear_threshold=0.22,
        mar_threshold=0.60,
        eye_closed_sec=1.5,
        yawn_sec=1.2,
        model_type="hybrid",  # "facemesh", "cnn", "hybrid", "haar"
        enable_sound=True
    ):
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.eye_closed_sec = eye_closed_sec
        self.yawn_sec = yawn_sec
        self.model_type = model_type

        # Real-time state
        self.eye_status = "Initializing..."
        self.yawn_status = "Normal"
        self.head_status = "Normal"
        self.ear = 0.0
        self.left_ear = 0.0
        self.right_ear = 0.0
        self.mar = 0.0
        self.head_pitch = 0.0
        self.head_roll = 0.0
        self.closed_duration = 0.0
        self.drowsiness_score = 0  # 0 to 100%

        # Counters & timing
        self.blink_count = 0
        self.yawn_count = 0
        self.nod_count = 0
        self._eye_was_closed = False
        self._yawn_start_time = None
        self._last_fps_time = time.time()
        self._fps_frame_count = 0
        self.fps = 0.0

        # CNN prediction throttling
        self._last_cnn_time = 0
        self._cnn_status = "Not checked"

        # Initialize audio alert
        self.alert = AlertSystem(eye_closed_sec=self.eye_closed_sec)
        self.alert.sound_enabled = enable_sound

        # Initialize vision backends
        self.face_landmarker = None
        self.cnn_model = None
        self.face_cascade = None
        self.eye_cascade = None

        self._init_models()

    def _find_file(self, filename, subdirs=("models", "assets")):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        roots = [
            os.path.abspath(os.path.join(base_dir, "..")),
            os.path.abspath(os.path.join(base_dir, "..", "..")),
            os.getcwd()
        ]
        for root in roots:
            for s in subdirs:
                p = os.path.join(root, s, filename)
                if os.path.isfile(p):
                    return p
            p2 = os.path.join(root, filename)
            if os.path.isfile(p2):
                return p2
        return None

    def _init_models(self):
        # 1. MediaPipe FaceLandmarker
        if MEDIAPIPE_AVAILABLE:
            task_path = self._find_file("face_landmarker.task", ["models", ""])
            if task_path and os.path.isfile(task_path):
                try:
                    base_options = python.BaseOptions(model_asset_path=task_path)
                    options = vision.FaceLandmarkerOptions(
                        base_options=base_options,
                        output_face_blendshapes=True,
                        output_facial_transformation_matrixes=True,
                        num_faces=1
                    )
                    self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
                except Exception as e:
                    self.face_landmarker = None

        # 2. CNN Model
        if KERAS_AVAILABLE:
            model_path = self._find_file("eye_model.keras", ["models", ""])
            if not model_path:
                model_path = self._find_file("drowsiness_model.keras", ["models", ""])
            if model_path:
                try:
                    self.cnn_model = keras.models.load_model(model_path)
                except Exception as e:
                    self.cnn_model = None

        # 3. Haar Cascades (Fallback)
        self.face_cascade = load_face_cascade()
        self.eye_cascade = load_eye_cascade()

    def _estimate_head_pose(self, landmarks, w, h):
        """
        Estimates Head Pitch, Yaw, Roll using key 3D facial landmarks.
        """
        try:
            # 3D model generic coordinates
            model_points = np.array([
                (0.0, 0.0, 0.0),             # Nose tip (1)
                (0.0, -330.0, -65.0),        # Chin (199)
                (-225.0, 170.0, -135.0),     # Left eye left corner (33)
                (225.0, 170.0, -135.0),      # Right eye right corner (263)
                (-150.0, -150.0, -125.0),    # Left Mouth corner (61)
                (150.0, -150.0, -125.0)      # Right mouth corner (291)
            ], dtype=np.float64)

            # 2D image coordinates from landmarks
            indices = [1, 199, 33, 263, 61, 291]
            image_points = np.array([
                (landmarks[idx].x * w, landmarks[idx].y * h) for idx in indices
            ], dtype=np.float64)

            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1))

            success, rotation_vector, _ = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success:
                return 0.0, 0.0

            rotation_mat, _ = cv2.Rodrigues(rotation_vector)
            pitch = math.atan2(rotation_mat[2, 1], rotation_mat[2, 2]) * 180 / math.pi
            roll = math.atan2(rotation_mat[1, 0], rotation_mat[0, 0]) * 180 / math.pi
            return float(pitch), float(roll)
        except Exception:
            return 0.0, 0.0

    def process_frame(self, frame):
        """
        Processes a single BGR video frame and returns (annotated_frame, metrics_dict).
        """
        now = time.time()
        self._fps_frame_count += 1
        if now - self._last_fps_time >= 1.0:
            self.fps = self._fps_frame_count / (now - self._last_fps_time)
            self._fps_frame_count = 0
            self._last_fps_time = now

        h, w = frame.shape[:2]
        annotated = frame.copy()

        face_detected = False
        is_closed = False
        is_yawning = False
        is_nodding = False

        # Mode A: MediaPipe FaceLandmarker
        if self.face_landmarker is not None and self.model_type != "haar":
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection = self.face_landmarker.detect(mp_img)

            if detection and detection.face_landmarks and len(detection.face_landmarks) > 0:
                face_detected = True
                landmarks = detection.face_landmarks[0]

                # 1. EAR Calculation
                left_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in self.LEFT_EYE_INDICES]
                right_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in self.RIGHT_EYE_INDICES]

                self.left_ear = calculate_ear(left_pts)
                self.right_ear = calculate_ear(right_pts)
                self.ear = (self.left_ear + self.right_ear) / 2.0

                # Draw eye outlines
                cv2.polylines(annotated, [np.array(left_pts)], True, (0, 255, 255), 1)
                cv2.polylines(annotated, [np.array(right_pts)], True, (0, 255, 255), 1)

                # 2. Check blendshapes if available
                blend_blink = 0.0
                if detection.face_blendshapes and len(detection.face_blendshapes) > 0:
                    blend_dict = {c.category_name: c.score for c in detection.face_blendshapes[0]}
                    b_l = blend_dict.get("eyeBlinkLeft", 0.0)
                    b_r = blend_dict.get("eyeBlinkRight", 0.0)
                    blend_blink = (b_l + b_r) / 2.0

                # EAR decision
                ear_closed = (self.ear < self.ear_threshold) or (blend_blink > 0.55)

                # 3. Optional CNN Verification
                cnn_closed = False
                if self.cnn_model is not None and (self.model_type in ["cnn", "hybrid"]):
                    if now - self._last_cnn_time > 0.15:
                        # Extract eye crop from landmark bounding box
                        all_eye_pts = left_pts + right_pts
                        ex, ey, ew, eh = cv2.boundingRect(np.array(all_eye_pts))
                        # expand slightly
                        pad_x, pad_y = int(ew * 0.2), int(eh * 0.2)
                        x1 = max(0, ex - pad_x)
                        y1 = max(0, ey - pad_y)
                        x2 = min(w, ex + ew + pad_x)
                        y2 = min(h, ey + eh + pad_y)

                        eye_crop = frame[y1:y2, x1:x2]
                        if eye_crop.size > 0:
                            pred = predict_eye_status(self.cnn_model, eye_crop)
                            self._cnn_status = pred
                            self._last_cnn_time = now

                    cnn_closed = (self._cnn_status == "Eyes Closed")

                if self.model_type == "cnn" and self.cnn_model is not None:
                    is_closed = cnn_closed
                elif self.model_type == "hybrid" and self.cnn_model is not None:
                    # Consensus or strong ear signal
                    is_closed = ear_closed or (cnn_closed and self.ear < (self.ear_threshold + 0.04))
                else:
                    is_closed = ear_closed

                # Blink counter
                if is_closed and not self._eye_was_closed:
                    self._eye_was_closed = True
                elif not is_closed and self._eye_was_closed:
                    self._eye_was_closed = False
                    self.blink_count += 1

                # 4. MAR Yawn Detection
                self.mar = calculate_mar_from_landmarks(landmarks, (h, w))
                mouth_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in [13, 14, 78, 308]]
                for pt in mouth_pts:
                    cv2.circle(annotated, pt, 2, (255, 0, 255), -1)

                if self.mar > self.mar_threshold:
                    if self._yawn_start_time is None:
                        self._yawn_start_time = now
                    elif now - self._yawn_start_time >= self.yawn_sec:
                        is_yawning = True
                        self.yawn_status = "YAWNING DETECTED!"
                else:
                    if self._yawn_start_time is not None:
                        if now - self._yawn_start_time >= self.yawn_sec:
                            self.yawn_count += 1
                        self._yawn_start_time = None
                    self.yawn_status = "Normal"

                # 5. Head Pose / Nodding Detection
                self.head_pitch, self.head_roll = self._estimate_head_pose(landmarks, w, h)
                if self.head_pitch < -16.0 or abs(self.head_roll) > 28.0:
                    is_nodding = True
                    self.head_status = "HEAD DROPPING / NODDING!"
                else:
                    self.head_status = "Normal"

        # Mode B: Fallback to Haar Cascade
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.18, minNeighbors=5, minSize=(80, 80)
            )

            if len(faces) > 0:
                face_detected = True
                fx, fy, fw, fh = faces[0]
                cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

                face_gray = gray[fy:fy + fh, fx:fx + fw]
                face_color = frame[fy:fy + fh, fx:fx + fw]

                eyes = detect_eyes(face_gray, self.eye_cascade)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(
                        annotated,
                        (fx + ex, fy + ey),
                        (fx + ex + ew, fy + ey + eh),
                        (255, 255, 0), 1
                    )

                if len(eyes) > 0:
                    ex, ey, ew, eh = eyes[0]
                    eye_crop = face_color[ey:ey + eh, ex:ex + ew]
                    pred = predict_eye_status(self.cnn_model, eye_crop)
                    is_closed = (pred == "Eyes Closed")
                else:
                    is_closed = False

                # Mouth
                mouth_box = detect_mouth(face_gray)
                if mouth_box:
                    mx, my, mw, mh = mouth_box
                    aspect = mh / max(1, mw)
                    if aspect > 0.65:
                        is_yawning = True
                        self.yawn_status = "YAWNING!"

        # Determine eye text label
        if not face_detected:
            self.eye_status = "Face Not Detected"
            self.closed_duration = 0.0
        elif is_closed:
            self.eye_status = "Eyes Closed"
        else:
            self.eye_status = "Eyes Open"

        # Update Alert System
        alert_result = self.alert.update(
            self.eye_status, is_yawning=is_yawning, is_nodding=is_nodding
        )
        self.closed_duration = alert_result.get("closed_duration", 0.0)
        alert_level = alert_result.get("alert_level", "NORMAL")

        # Compute Drowsiness Score (0% to 100%)
        drowsy_ratio = min(1.0, self.closed_duration / max(0.1, self.eye_closed_sec))
        score = int(drowsy_ratio * 70)
        if is_yawning:
            score += 20
        if is_nodding:
            score += 15
        if self.yawn_count >= 3:
            score += 15
        self.drowsiness_score = min(100, score)

        # Draw HUD overlays on frame
        self._draw_hud(annotated, alert_level)

        metrics = {
            "face_detected": face_detected,
            "eye_status": self.eye_status,
            "is_closed": is_closed,
            "ear": self.ear,
            "mar": self.mar,
            "closed_duration": self.closed_duration,
            "yawn_status": self.yawn_status,
            "head_status": self.head_status,
            "drowsiness_score": self.drowsiness_score,
            "blink_count": self.blink_count,
            "yawn_count": self.yawn_count,
            "alert_level": alert_level,
            "alarm_playing": self.alert.alarm_playing,
            "fps": self.fps
        }

        return annotated, metrics

    def _draw_hud(self, img, alert_level):
        """Draws clean, professional HUD graphics on the frame."""
        h, w = img.shape[:2]

        # Top banner based on alert level
        if alert_level == "CRITICAL":
            # Flashing banner
            color = (0, 0, 220)
            text = f"CRITICAL: DRIVER SLEEPING! ({self.closed_duration:.1f}s)"
            cv2.rectangle(img, (0, 0), (w, 55), color, -1)
            cv2.putText(img, text, (20, 38), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        elif alert_level == "WARNING":
            color = (0, 140, 255)  # Orange
            warning_msg = "WARNING: DROWSINESS DETECTED"
            if self.yawn_status != "Normal":
                warning_msg = "WARNING: YAWNING DETECTED!"
            elif self.head_status != "Normal":
                warning_msg = "WARNING: HEAD NODDING DETECTED!"
            cv2.rectangle(img, (0, 0), (w, 50), color, -1)
            cv2.putText(img, warning_msg, (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)
        else:
            # Semi-transparent dark header
            header = img[0:45, 0:w].copy()
            cv2.rectangle(header, (0, 0), (w, 45), (20, 20, 20), -1)
            cv2.addWeighted(header, 0.6, img[0:45, 0:w], 0.4, 0, img[0:45, 0:w])

            eye_color = (0, 255, 0) if self.eye_status == "Eyes Open" else (0, 165, 255)
            cv2.putText(
                img, f"Status: {self.eye_status}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, eye_color, 2
            )
            cv2.putText(
                img, f"FPS: {self.fps:.1f}", (w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2
            )

        # Bottom telemetry bar
        bar_y = h - 45
        footer = img[bar_y:h, 0:w].copy()
        cv2.rectangle(footer, (0, 0), (w, 45), (15, 15, 15), -1)
        cv2.addWeighted(footer, 0.75, img[bar_y:h, 0:w], 0.25, 0, img[bar_y:h, 0:w])

        stats_text = (
            f"EAR: {self.ear:.2f} (Thresh: {self.ear_threshold})  |  "
            f"MAR: {self.mar:.2f}  |  "
            f"Blinks: {self.blink_count}  |  "
            f"Yawns: {self.yawn_count}"
        )
        cv2.putText(img, stats_text, (15, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        # Drowsiness meter bar on top-right
        meter_w = 160
        meter_h = 16
        meter_x = w - meter_w - 20
        meter_y = 65

        # Background
        cv2.rectangle(img, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (50, 50, 50), -1)
        # Fill
        fill_w = int((self.drowsiness_score / 100.0) * meter_w)
        fill_color = (0, 255, 0) if self.drowsiness_score < 40 else ((0, 165, 255) if self.drowsiness_score < 70 else (0, 0, 255))
        cv2.rectangle(img, (meter_x, meter_y), (meter_x + fill_w, meter_y + meter_h), fill_color, -1)
        cv2.rectangle(img, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (200, 200, 200), 1)

        cv2.putText(
            img, f"Fatigue: {self.drowsiness_score}%", (meter_x, meter_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1
        )

    def close(self):
        """Cleanly releases resources."""
        if self.alert:
            self.alert.stop()
        if self.face_landmarker:
            try:
                self.face_landmarker.close()
            except Exception:
                pass
            self.face_landmarker = None
