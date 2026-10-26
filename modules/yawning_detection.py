import cv2
import numpy as np


def load_face_cascade():
    """Loads OpenCV Haar cascade for face detection."""
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(path)


def detect_mouth(face_gray):
    """
    Detects mouth in the lower half of the face.
    Returns (x, y, w, h) or None.
    """
    if face_gray is None or face_gray.size == 0:
        return None

    height, width = face_gray.shape
    lower_face = face_gray[height // 2:, :]

    mouth_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_smile.xml"
    )

    mouths = mouth_cascade.detectMultiScale(
        lower_face,
        scaleFactor=1.5,
        minNeighbors=15,
        minSize=(30, 20)
    )

    if len(mouths) > 0:
        # Largest detected feature in lower face
        x, y, w, h = max(mouths, key=lambda m: m[2] * m[3])
        return (x, y + height // 2, w, h)

    # Fallback anatomical mouth region
    mouth_w = int(width * 0.5)
    mouth_h = int(height * 0.25)
    mouth_x = int(width * 0.25)
    mouth_y = int(height * 0.65)
    return (mouth_x, mouth_y, mouth_w, mouth_h)


def calculate_mar(inner_top, inner_bottom, left_corner, right_corner):
    """
    Calculates Mouth Aspect Ratio (MAR) using landmark points:
    MAR = |inner_top - inner_bottom| / |left_corner - right_corner|
    Normal mouth: 0.15 - 0.35
    Yawning mouth: > 0.55 - 0.65
    """
    try:
        vertical = np.linalg.norm(np.array(inner_top) - np.array(inner_bottom))
        horizontal = np.linalg.norm(np.array(left_corner) - np.array(right_corner))
        if horizontal == 0:
            return 0.0
        return float(vertical / horizontal)
    except Exception:
        return 0.0


def calculate_mar_from_landmarks(landmarks, image_shape):
    """
    Calculates MAR using MediaPipe FaceLandmarker indices:
    - Inner top lip: 13
    - Inner bottom lip: 14
    - Left mouth corner: 78
    - Right mouth corner: 308
    """
    h, w = image_shape[:2]
    try:
        p13 = (landmarks[13].x * w, landmarks[13].y * h)
        p14 = (landmarks[14].x * w, landmarks[14].y * h)
        p78 = (landmarks[78].x * w, landmarks[78].y * h)
        p308 = (landmarks[308].x * w, landmarks[308].y * h)
        return calculate_mar(p13, p14, p78, p308)
    except Exception:
        return 0.0
