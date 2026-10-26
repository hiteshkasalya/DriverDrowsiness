import sys
import os
sys.path.append(os.path.abspath(".."))

import streamlit as st
import cv2
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

from modules.eye_detection import load_eye_cascade, detect_eyes, predict_eye_status
from modules.yawning_detection import load_face_cascade, detect_mouth
from modules.alert_system import AlertSystem

import tensorflow as tf

st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")
st.title("Driver Drowsiness Detection System")
st.write("CNN-based driver monitoring system")
st.info("Allow camera access and keep your face clearly visible.")

model = tf.keras.models.load_model("eye_model.keras")
eye_cascade = load_eye_cascade()
face_cascade = load_face_cascade()


class DrowsinessDetector(VideoTransformerBase):

    def __init__(self):
        self.eye_status = "Checking..."
        self.last_prediction_time = 0
        self.alert = None
        self.yawn_status = "Not checked"

        try:
            self.alert = AlertSystem("../assets/alarm.mp3")
        except Exception as e:
            print("Alarm error:", e)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (640, 480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2,
            minNeighbors=5, minSize=(60, 60)
        )

        if len(faces) == 0:
            self.eye_status = "Face not detected"
            self.yawn_status = "Not checked"

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            face_gray = gray[y:y+h, x:x+w]
            face = img[y:y+h, x:x+w]

            eyes = detect_eyes(face_gray, eye_cascade)

            if len(eyes) > 0 and time.time() - self.last_prediction_time > 0.5:
                ex, ey, ew, eh = eyes[0]
                eye_crop = face[ey:ey+eh, ex:ex+ew]

                self.eye_status = predict_eye_status(model, eye_crop)
                self.last_prediction_time = time.time()

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(img, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 255, 0), 2)

            mouth = detect_mouth(face_gray)

            if mouth:
                mx, my, mw, mh = mouth
                cv2.rectangle(
                    img, (x+mx, y+my), (x+mx+mw, y+my+mh),
                    (255, 0, 255), 2
                )
                self.yawn_status = "Mouth detected"

        if self.alert:
            self.alert.update(self.eye_status)

        if self.eye_status == "Eyes Open":
            eye_color = (0, 255, 0)

        elif self.eye_status == "Eyes Closed":
            eye_color = (0, 0, 255)

        else:
            eye_color = (0, 255, 255)

        cv2.putText(
            img, self.eye_status, (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, eye_color, 2
        )
        cv2.putText(
            img, "Yawning: " + self.yawn_status, (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2
        )

        return img


webrtc_streamer(
    key="drowsiness-camera",
    video_transformer_factory=DrowsinessDetector,
    media_stream_constraints={"video": True, "audio": False}
)
