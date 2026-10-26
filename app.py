import os
import sys
import time
import cv2
import av
import numpy as np
import streamlit as st

# Add project root and minipro directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

from modules.drowsiness_detector import ComprehensiveDrowsinessDetector

st.set_page_config(
    page_title="Driver Drowsiness Detection System",
    page_icon="🚗",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e2130;
        padding: 18px;
        border-radius: 10px;
        border-left: 5px solid #00d26a;
        margin-bottom: 12px;
        color: #f1f1f1;
    }
    .metric-card-danger {
        background-color: #3b1111;
        padding: 18px;
        border-radius: 10px;
        border-left: 5px solid #ff4757;
        margin-bottom: 12px;
        color: #f1f1f1;
    }
    .metric-card-warning {
        background-color: #38270a;
        padding: 18px;
        border-radius: 10px;
        border-left: 5px solid #ffa502;
        margin-bottom: 12px;
        color: #f1f1f1;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚗 Intelligent Driver Drowsiness Detection System")
st.caption("Real-Time Multi-Modal Monitoring with MediaPipe FaceMesh, EAR/MAR Geometry & CNN Deep Learning")

# Sidebar Configuration
st.sidebar.header("⚙️ Detection Settings")

backend_mode = st.sidebar.selectbox(
    "Detection Mode",
    options=["hybrid", "facemesh", "cnn", "haar"],
    index=0,
    format_func=lambda x: {
        "hybrid": "Hybrid (FaceMesh + CNN Consensus)",
        "facemesh": "MediaPipe FaceMesh (Ultra-Fast EAR/MAR)",
        "cnn": "Deep Learning CNN Only",
        "haar": "Haar Cascade (Legacy Fallback)"
    }[x]
)

eye_closed_sec = st.sidebar.slider(
    "Eye Closed Alarm Threshold (seconds)",
    min_value=1.0, max_value=3.0, value=1.5, step=0.1,
    help="Duration driver eyes must remain closed before triggering the critical alarm."
)

ear_threshold = st.sidebar.slider(
    "EAR Closure Sensitivity",
    min_value=0.15, max_value=0.30, value=0.22, step=0.01,
    help="Higher values make eye closure detection more sensitive."
)

mar_threshold = st.sidebar.slider(
    "Mouth Aspect Ratio (Yawn) Threshold",
    min_value=0.45, max_value=0.75, value=0.60, step=0.02,
    help="Threshold for detecting yawning mouth openness."
)

enable_sound = st.sidebar.checkbox("Enable Audio Alarm", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🖥️ Native Desktop High-FPS App")
st.sidebar.info("For max 60+ FPS and zero browser lag, you can run the standalone desktop app:")
st.sidebar.code("python run_detector.py", language="bash")

# Video Input Source Selection
input_mode = st.radio(
    "Select Video Source:",
    ["Live Native Camera (Highest FPS)", "Browser WebRTC Camera", "Upload Video File"],
    horizontal=True
)

# ----------------- Mode 1: Live Native Camera -----------------
if input_mode == "Live Native Camera (Highest FPS)":
    st.write("Streams directly from your connected camera with real-time HUD telemetry and sound alerts.")
    col1, col2 = st.columns([3, 1])

    with col1:
        run_cam = st.toggle("Start Camera Feed", value=False)
        frame_placeholder = st.empty()

    with col2:
        status_card = st.empty()
        fatigue_card = st.empty()
        telemetry_card = st.empty()

    if run_cam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)

        if not cap.isOpened():
            st.error("Could not access webcam. Please check your camera connection or permissions.")
        else:
            detector = ComprehensiveDrowsinessDetector(
                ear_threshold=ear_threshold,
                mar_threshold=mar_threshold,
                eye_closed_sec=eye_closed_sec,
                model_type=backend_mode,
                enable_sound=enable_sound
            )

            try:
                while run_cam:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        time.sleep(0.03)
                        continue

                    frame = cv2.flip(frame, 1)
                    annotated, metrics = detector.process_frame(frame)

                    # Update video image
                    frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                    # Update sidebar stats
                    alert_level = metrics["alert_level"]
                    if alert_level == "CRITICAL":
                        status_card.markdown(f"""
                        <div class="metric-card-danger">
                            <h3>🚨 SLEEPING ALERT!</h3>
                            <p>Eyes closed: <b>{metrics['closed_duration']:.1f}s</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif alert_level == "WARNING":
                        status_card.markdown(f"""
                        <div class="metric-card-warning">
                            <h3>⚠️ DROWSY WARNING</h3>
                            <p>{metrics['yawn_status'] if metrics['yawn_status'] != 'Normal' else metrics['head_status']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        status_card.markdown(f"""
                        <div class="metric-card">
                            <h3>✅ DRIVER AWAKE</h3>
                            <p>Status: {metrics['eye_status']}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    fatigue_card.metric("Fatigue Score", f"{metrics['drowsiness_score']}%")
                    telemetry_card.markdown(f"""
                    **Telemetry:**
                    - EAR: `{metrics['ear']:.2f}`
                    - MAR: `{metrics['mar']:.2f}`
                    - Blinks: `{metrics['blink_count']}`
                    - Yawns: `{metrics['yawn_count']}`
                    - FPS: `{metrics['fps']:.1f}`
                    """)

                    time.sleep(0.01)
            finally:
                cap.release()
                detector.close()

# ----------------- Mode 2: Browser WebRTC Camera -----------------
elif input_mode == "Browser WebRTC Camera":
    st.write("Streams through your browser via WebRTC.")
    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

        class WebRTCProcessor(VideoProcessorBase):
            def __init__(self):
                self.detector = ComprehensiveDrowsinessDetector(
                    ear_threshold=ear_threshold,
                    mar_threshold=mar_threshold,
                    eye_closed_sec=eye_closed_sec,
                    model_type=backend_mode,
                    enable_sound=enable_sound
                )

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                annotated, _ = self.detector.process_frame(img)
                return av.VideoFrame.from_ndarray(annotated, format="bgr24")

            def __del__(self):
                if hasattr(self, "detector"):
                    self.detector.close()

        webrtc_streamer(
            key="drowsiness-detector-webrtc",
            video_processor_factory=WebRTCProcessor,
            media_stream_constraints={"video": True, "audio": False}
        )
    except Exception as e:
        st.error(f"WebRTC initialization error: {e}")

# ----------------- Mode 3: Upload Video File -----------------
else:
    st.write("Upload a pre-recorded driving video clip (.mp4, .avi, .mov) to test detection.")
    uploaded_file = st.file_uploader("Upload driving video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        import tempfile
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        detector = ComprehensiveDrowsinessDetector(
            ear_threshold=ear_threshold,
            mar_threshold=mar_threshold,
            eye_closed_sec=eye_closed_sec,
            model_type=backend_mode,
            enable_sound=enable_sound
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            video_placeholder = st.empty()
        with col2:
            stats_box = st.empty()

        start_btn = st.button("Analyze Video Clip")
        if start_btn:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                annotated, metrics = detector.process_frame(frame)
                frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                stats_box.markdown(f"""
                ### Real-Time Stats
                - **Eye Status**: {metrics['eye_status']}
                - **Closed Duration**: {metrics['closed_duration']:.1f}s
                - **EAR**: {metrics['ear']:.2f}
                - **MAR**: {metrics['mar']:.2f}
                - **Alert Level**: `{metrics['alert_level']}`
                - **Fatigue**: `{metrics['drowsiness_score']}%`
                """)
                time.sleep(0.02)

            cap.release()
            detector.close()
            st.success("Video analysis completed!")
