import cv2
import numpy as np
from collections import deque

# Local deque per default fallback
_default_eye_history = deque(maxlen=3)


def load_eye_cascade():
    """Loads OpenCV Haar cascade for eye detection with fallback."""
    path = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        path = cv2.data.haarcascades + "haarcascade_eye.xml"
        cascade = cv2.CascadeClassifier(path)
    return cascade


def detect_eyes(face_gray, eye_cascade):
    """
    Detects eyes inside a cropped grayscale face.
    If Haar cascade fails (e.g., eyes closed), falls back to anatomical eye regions.
    """
    eyes = eye_cascade.detectMultiScale(
        face_gray,
        scaleFactor=1.15,
        minNeighbors=4,
        minSize=(24, 24)
    )

    if len(eyes) >= 2:
        # Sort left to right by x coordinate
        eyes = sorted(eyes, key=lambda e: e[0])[:2]
        return eyes
    elif len(eyes) == 1:
        return eyes

    # Anatomical fallback: Eyes are in the upper half of the face
    # This prevents total detection loss when driver closes their eyes!
    h, w = face_gray.shape
    eye_h = int(h * 0.25)
    eye_w = int(w * 0.30)
    eye_y = int(h * 0.22)

    left_x = int(w * 0.15)
    right_x = int(w * 0.55)

    fallback_eyes = [
        (left_x, eye_y, eye_w, eye_h),
        (right_x, eye_y, eye_w, eye_h)
    ]
    return fallback_eyes


def _model_has_rescaling_layer(model):
    """Checks if the Keras model already contains an internal Rescaling layer."""
    if model is None:
        return False
    try:
        for layer in getattr(model, "layers", []):
            if "rescaling" in layer.name.lower():
                return True
    except Exception:
        pass
    return False


def predict_eye_status(model, eye_crop, history=None):
    """
    Predicts if eye is Open or Closed using the CNN model.
    Fixes the critical double-normalization bug.
    """
    if eye_crop is None or eye_crop.size == 0 or model is None:
        return "Unknown"

    try:
        eye = cv2.resize(eye_crop, (64, 64))
        if len(eye.shape) == 2:
            eye = cv2.cvtColor(eye, cv2.COLOR_GRAY2RGB)
        else:
            eye = cv2.cvtColor(eye, cv2.COLOR_BGR2RGB)

        eye = eye.astype("float32")

        # Avoid double-normalization: check if model has internal rescaling layer
        if not _model_has_rescaling_layer(model):
            eye = eye / 255.0

        eye = np.expand_dims(eye, axis=0)
        prediction = model.predict(eye, verbose=0)

        # Check output shape: binary sigmoid [0..1] vs multi-class softmax
        if prediction.shape[-1] == 1:
            score = float(prediction[0][0])
            # For eye_model.keras: >= 0.5 is Open, < 0.5 is Closed
            status = "Eyes Open" if score >= 0.5 else "Eyes Closed"
        elif prediction.shape[-1] == 4:
            # drowsiness_model.keras: ['Closed': 0, 'Open': 1, 'no_yawn': 2, 'yawn': 3]
            class_idx = int(np.argmax(prediction[0]))
            status = "Eyes Closed" if class_idx == 0 else "Eyes Open"
        else:
            status = "Eyes Open"

        target_history = history if history is not None else _default_eye_history
        target_history.append(status)

        # Immediate closure detection: if latest 2 or majority are closed, return closed
        closed_count = target_history.count("Eyes Closed")
        if closed_count >= 2:
            return "Eyes Closed"
        return "Eyes Open"

    except Exception as e:
        return "Eyes Open"


def calculate_ear(eye_points):
    """
    Calculates Eye Aspect Ratio (EAR) given 6 (x, y) landmark coordinates:
    p1: outer corner
    p2, p3: top eyelid
    p4: inner corner
    p5, p6: bottom eyelid
    EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
    """
    try:
        p1, p2, p3, p4, p5, p6 = eye_points
        # Vertical distances
        v1 = np.linalg.norm(np.array(p2) - np.array(p6))
        v2 = np.linalg.norm(np.array(p3) - np.array(p5))
        # Horizontal distance
        h = np.linalg.norm(np.array(p1) - np.array(p4))

        if h == 0:
            return 0.0

        ear = (v1 + v2) / (2.0 * h)
        return float(ear)
    except Exception:
        return 0.0
