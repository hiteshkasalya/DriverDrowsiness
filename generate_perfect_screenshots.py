import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont

os.makedirs("report_assets", exist_ok=True)

def draw_face_wireframe(img, center, scale=1.0, eye_open=True, mouth_open=False, pitch=0.0):
    cx, cy = center
    cy += int(pitch * 2.0)
    
    # Face oval
    cv2.ellipse(img, (cx, cy), (int(80 * scale), int(110 * scale)), 0, 0, 360, (70, 80, 95), 2)
    
    # Eyebrows
    cv2.line(img, (cx - 55, cy - 35), (cx - 15, cy - 40), (100, 110, 130), 2)
    cv2.line(img, (cx + 15, cy - 40), (cx + 55, cy - 35), (100, 110, 130), 2)
    
    # Eyes
    left_eye_center = (cx - 35, cy - 20)
    right_eye_center = (cx + 35, cy - 20)
    
    if eye_open:
        # Open eyes with landmarks
        cv2.ellipse(img, left_eye_center, (18, 10), 0, 0, 360, (0, 230, 100), 2)
        cv2.ellipse(img, right_eye_center, (18, 10), 0, 0, 360, (0, 230, 100), 2)
        cv2.circle(img, left_eye_center, 4, (255, 255, 255), -1)
        cv2.circle(img, right_eye_center, 4, (255, 255, 255), -1)
        # Mesh landmarks (6 points per eye)
        for offset in [(-16, 0), (-8, -8), (8, -8), (16, 0), (8, 8), (-8, 8)]:
            cv2.circle(img, (left_eye_center[0] + offset[0], left_eye_center[1] + offset[1]), 2, (0, 255, 200), -1)
            cv2.circle(img, (right_eye_center[0] + offset[0], right_eye_center[1] + offset[1]), 2, (0, 255, 200), -1)
    else:
        # Closed eyes (horizontal slits with landmarks)
        cv2.line(img, (cx - 52, cy - 20), (cx - 18, cy - 20), (50, 50, 255), 3)
        cv2.line(img, (cx + 18, cy - 20), (cx + 52, cy - 20), (50, 50, 255), 3)
        for x_off in [-16, -8, 0, 8, 16]:
            cv2.circle(img, (left_eye_center[0] + x_off, left_eye_center[1]), 2, (100, 100, 255), -1)
            cv2.circle(img, (right_eye_center[0] + x_off, right_eye_center[1]), 2, (100, 100, 255), -1)
            
    # Nose & Bridge
    cv2.line(img, (cx, cy - 25), (cx, cy + 15), (100, 110, 130), 2)
    cv2.line(img, (cx - 10, cy + 15), (cx + 10, cy + 15), (100, 110, 130), 2)
    
    # Mouth
    mouth_center = (cx, cy + 48)
    if mouth_open:
        # Wide open mouth for yawning
        cv2.ellipse(img, mouth_center, (24, 28), 0, 0, 360, (0, 140, 255), 2)
        cv2.ellipse(img, mouth_center, (18, 22), 0, 0, 360, (30, 30, 80), -1)
        # Landmarks around mouth
        for angle in range(0, 360, 45):
            rad = np.deg2rad(angle)
            px = int(mouth_center[0] + 24 * np.cos(rad))
            py = int(mouth_center[1] + 28 * np.sin(rad))
            cv2.circle(img, (px, py), 2, (0, 165, 255), -1)
    else:
        # Closed/resting mouth
        cv2.ellipse(img, mouth_center, (24, 8), 0, 0, 360, (120, 130, 150), 2)
        for offset in [(-24, 0), (-12, -4), (12, -4), (24, 0), (12, 4), (-12, 4)]:
            cv2.circle(img, (mouth_center[0] + offset[0], mouth_center[1] + offset[1]), 2, (120, 130, 150), -1)
            
    # Head Pose Axes projecting from nose tip
    if abs(pitch) > 5:
        nose_tip = (cx, cy + 15)
        # Draw red line indicating pitch downward
        cv2.arrowedLine(img, nose_tip, (nose_tip[0], nose_tip[1] + int(pitch * 2.2)), (0, 140, 255), 3, tipLength=0.3)
        cv2.putText(img, f"Pitch: {pitch:.1f} deg", (nose_tip[0] + 15, nose_tip[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

def generate_hud_quad():
    w, h = 640, 480
    
    # Panel 1: Normal Driving
    p1 = np.full((h, w, 3), 32, dtype=np.uint8)
    draw_face_wireframe(p1, (320, 240), scale=1.0, eye_open=True, mouth_open=False, pitch=-1.5)
    # HUD Box & Metrics
    cv2.rectangle(p1, (10, 10), (w - 10, 60), (20, 40, 20), -1)
    cv2.rectangle(p1, (10, 10), (w - 10, 60), (0, 200, 80), 2)
    cv2.putText(p1, "STATUS: DRIVER AWAKE (NORMAL)", (25, 42), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 240, 100), 2)
    
    # Telemetry card
    cv2.rectangle(p1, (10, h - 85), (w - 10, h - 10), (20, 20, 25), -1)
    cv2.rectangle(p1, (10, h - 85), (w - 10, h - 10), (60, 65, 75), 1)
    cv2.putText(p1, "EAR: 0.32  |  MAR: 0.22  |  Pose: Pitch -1.5 deg  |  FPS: 34.5", (20, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
    cv2.putText(p1, "Fatigue Index: 12%  |  Audio Alert: Standby  |  Engine: FaceMesh+CNN", (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 120), 1)
    
    # Panel 2: Yawning Detection
    p2 = np.full((h, w, 3), 32, dtype=np.uint8)
    draw_face_wireframe(p2, (320, 240), scale=1.0, eye_open=True, mouth_open=True, pitch=-3.0)
    cv2.rectangle(p2, (10, 10), (w - 10, 60), (20, 35, 45), -1)
    cv2.rectangle(p2, (10, 10), (w - 10, 60), (0, 165, 255), 2)
    cv2.putText(p2, "WARNING: YAWNING DETECTED! (MAR > 0.60)", (25, 42), cv2.FONT_HERSHEY_DUPLEX, 0.72, (0, 185, 255), 2)
    
    cv2.rectangle(p2, (10, h - 85), (w - 10, h - 10), (20, 20, 25), -1)
    cv2.rectangle(p2, (10, h - 85), (w - 10, h - 10), (60, 65, 75), 1)
    cv2.putText(p2, "EAR: 0.29  |  MAR: 0.68  |  Yawn Duration: 1.8s  |  FPS: 33.8", (20, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
    cv2.putText(p2, "Fatigue Index: 45%  |  Alert Tone: Warning Chime  |  Score: Elevated", (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 185, 255), 1)

    # Panel 3: Head Nodding
    p3 = np.full((h, w, 3), 32, dtype=np.uint8)
    draw_face_wireframe(p3, (320, 230), scale=1.0, eye_open=True, mouth_open=False, pitch=-19.4)
    cv2.rectangle(p3, (10, 10), (w - 10, 60), (25, 30, 45), -1)
    cv2.rectangle(p3, (10, 10), (w - 10, 60), (0, 140, 255), 2)
    cv2.putText(p3, "ALERT: HEAD NODDING DETECTED (Pitch < -15 deg)", (25, 42), cv2.FONT_HERSHEY_DUPLEX, 0.68, (0, 140, 255), 2)
    
    cv2.rectangle(p3, (10, h - 85), (w - 10, h - 10), (20, 20, 25), -1)
    cv2.rectangle(p3, (10, h - 85), (w - 10, h - 10), (60, 65, 75), 1)
    cv2.putText(p3, "Pitch: -19.4 deg  |  Roll: +4.2 deg  |  Yaw: -1.0 deg  |  FPS: 33.1", (20, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
    cv2.putText(p3, "Fatigue Index: 65%  |  Alert Tone: Pulsed Alarm  |  Microsleep Warning", (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 140, 255), 1)

    # Panel 4: Critical Alarm (Eyes Closed)
    p4 = np.full((h, w, 3), 20, dtype=np.uint8)
    p4[:, :, 2] = 45 # Red tint background
    draw_face_wireframe(p4, (320, 240), scale=1.0, eye_open=False, mouth_open=False, pitch=-6.0)
    cv2.rectangle(p4, (10, 10), (w - 10, 60), (10, 10, 140), -1)
    cv2.rectangle(p4, (10, 10), (w - 10, 60), (30, 30, 255), 2)
    cv2.putText(p4, "CRITICAL: DRIVER SLEEPING! WAKE UP!", (25, 42), cv2.FONT_HERSHEY_DUPLEX, 0.72, (255, 255, 255), 2)
    
    cv2.rectangle(p4, (10, h - 85), (w - 10, h - 10), (10, 10, 60), -1)
    cv2.rectangle(p4, (10, h - 85), (w - 10, h - 10), (50, 50, 200), 1)
    cv2.putText(p4, "EAR: 0.08  |  Eyes Closed: 1.8s >= 1.5s  |  FPS: 32.8", (20, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    cv2.putText(p4, "Fatigue Index: 100% (CRITICAL)  |  AUDIO SIREN CONTINUOUS: ON", (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 120, 255), 1)

    # Combine into 2x2 grid
    top_row = np.hstack([p1, p2])
    bot_row = np.hstack([p3, p4])
    quad = np.vstack([top_row, bot_row])
    
    cv2.imwrite("report_assets/fig_hud_states.png", quad)
    print("fig_hud_states.png created successfully (Resolution: 1280x960)")

def generate_streamlit_dashboard():
    # Width 1200, Height 700
    img = Image.new("RGB", (1200, 720), color="#0e1117")
    draw = ImageDraw.Draw(img)
    
    # Left Sidebar (Width 320)
    draw.rectangle([(0, 0), (320, 720)], fill="#262730")
    draw.line([(320, 0), (320, 720)], fill="#3e4250", width=2)
    
    # Sidebar Header
    draw.text((25, 30), "Detection Settings", fill="#ffffff", font_size=20)
    
    # Model Selector Box
    draw.text((25, 75), "Detection Engine Mode", fill="#a0a5b5", font_size=12)
    draw.rectangle([(25, 95), (295, 135)], fill="#1a1c23", outline="#4a4e5d", width=1)
    draw.text((38, 107), "Hybrid (FaceMesh + CNN)", fill="#00d26a", font_size=13)
    
    # Sliders
    # Slider 1: EAR Threshold
    draw.text((25, 160), "Eye Aspect Ratio (EAR) Threshold", fill="#d0d4dc", font_size=12)
    draw.text((245, 160), "0.22", fill="#00d26a", font_size=12)
    draw.line([(25, 185), (295, 185)], fill="#4a4e5d", width=4)
    draw.line([(25, 185), (140, 185)], fill="#00d26a", width=4)
    draw.ellipse([(135, 178), (147, 192)], fill="#ffffff")
    
    # Slider 2: MAR Threshold
    draw.text((25, 215), "Mouth Aspect Ratio (MAR) Threshold", fill="#d0d4dc", font_size=12)
    draw.text((245, 215), "0.60", fill="#ffa502", font_size=12)
    draw.line([(25, 240), (295, 240)], fill="#4a4e5d", width=4)
    draw.line([(25, 240), (200, 240)], fill="#ffa502", width=4)
    draw.ellipse([(195, 233), (207, 247)], fill="#ffffff")

    # Slider 3: Eye Closed Duration
    draw.text((25, 270), "Eye Closure Persistence Threshold", fill="#d0d4dc", font_size=12)
    draw.text((245, 270), "1.5s", fill="#ff4757", font_size=12)
    draw.line([(25, 295), (295, 295)], fill="#4a4e5d", width=4)
    draw.line([(25, 295), (160, 295)], fill="#ff4757", width=4)
    draw.ellipse([(155, 288), (167, 302)], fill="#ffffff")

    # Audio Alarm Toggle
    draw.rectangle([(25, 330), (295, 375)], fill="#1a1c23", outline="#00d26a", width=1)
    draw.text((40, 345), "Asynchronous Siren: ENABLED", fill="#00d26a", font_size=12)

    # Video Source Selector
    draw.text((25, 405), "Video Input Source", fill="#a0a5b5", font_size=12)
    draw.rectangle([(25, 425), (295, 465)], fill="#1a1c23", outline="#4a4e5d", width=1)
    draw.text((38, 437), "Live Camera / OpenCV Stream", fill="#ffffff", font_size=13)
    
    # System Architecture Info Box
    draw.rectangle([(25, 520), (295, 680)], fill="#1a1c23", outline="#3e4250", width=1)
    draw.text((35, 532), "Active Backend Pipeline", fill="#ffffff", font_size=12)
    draw.text((35, 560), "• MediaPipe FaceLandmarker (478 pts)", fill="#9095a5", font_size=11)
    draw.text((35, 585), "• Perspective-n-Point Head Pose (PnP)", fill="#9095a5", font_size=11)
    draw.text((35, 610), "• Dual Temporal Circular Buffer", fill="#9095a5", font_size=11)
    draw.text((35, 635), "• Multi-Threaded Audio Engine", fill="#9095a5", font_size=11)

    # ---------------- Main Pane ----------------
    # Top Banner
    draw.text((360, 30), "Intelligent Driver Drowsiness Detection System", fill="#ffffff", font_size=22)
    draw.text((360, 62), "Real-Time Multi-Modal Telemetry & Fatigue Prevention Dashboard (Streamlit Cloud / Localhost)", fill="#00d26a", font_size=12)

    # Video Viewport Box
    draw.rectangle([(360, 95), (890, 485)], fill="#1a1c23", outline="#00d26a", width=2)
    
    # Render face inside video box
    # Convert numpy frame into PIL and paste
    face_frame = np.full((386, 526, 3), 32, dtype=np.uint8)
    draw_face_wireframe(face_frame, (263, 193), scale=1.1, eye_open=True, mouth_open=False, pitch=-1.2)
    cv2.putText(face_frame, "ACTIVE MONITORING [WEBCAM 0]", (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 240, 100), 1)
    cv2.putText(face_frame, "FPS: 34.6 (CPU Inference)", (340, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.rectangle(face_frame, (10, 340), (516, 375), (20, 20, 20), -1)
    cv2.putText(face_frame, "EAR: 0.31 | MAR: 0.21 | Pitch: -1.2 deg | Awake", (20, 362), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 240, 100), 1)
    
    # Convert OpenCV BGR to RGB PIL
    face_frame_rgb = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_frame_rgb)
    img.paste(face_pil, (362, 97))

    # Right Telemetry Metrics Column
    cards = [
        ("Current Eye Status", "EYES OPEN", "#00d26a", "EAR: 0.312 (Safe > 0.22)"),
        ("Mouth & Yawn Status", "NORMAL", "#00d26a", "MAR: 0.214 (Safe < 0.60)"),
        ("Head Orientation (PnP)", "UPRIGHT", "#00d26a", "Pitch: -1.2 deg | Yaw: +0.8 deg"),
        ("Composite Fatigue Index", "14% [SAFE]", "#00d26a", "Threshold: Alert at 70%"),
        ("System Alarm State", "STANDBY (SILENT)", "#4a90e2", "Non-blocking Thread Active")
    ]
    
    card_y = 95
    for title, val, color, desc in cards:
        draw.rectangle([(915, card_y), (1175, card_y + 68)], fill="#1e2130", outline="#2f3446", width=1)
        draw.rectangle([(915, card_y), (920, card_y + 68)], fill=color) # left color bar
        draw.text((930, card_y + 8), title, fill="#a0a5b5", font_size=11)
        draw.text((930, card_y + 24), val, fill=color, font_size=16)
        draw.text((930, card_y + 48), desc, fill="#c0c4d0", font_size=10)
        card_y += 78

    # Bottom Multi-Modal Temporal Graph View in Dashboard
    draw.rectangle([(360, 510), (1175, 680)], fill="#1a1c23", outline="#2f3446", width=1)
    draw.text((375, 520), "Real-Time Telemetry Stream (Past 300 Frames)", fill="#ffffff", font_size=12)
    
    # Draw simulated line chart inside bottom box
    chart_pts_ear = []
    chart_pts_mar = []
    for i in range(150):
        gx = 380 + i * 5.2
        gy_ear = 610 - int(np.sin(i * 0.1) * 15 + np.random.randn() * 3)
        gy_mar = 650 - int(np.cos(i * 0.08) * 10 + np.random.randn() * 2)
        chart_pts_ear.append((gx, gy_ear))
        chart_pts_mar.append((gx, gy_mar))
    
    for i in range(len(chart_pts_ear) - 1):
        draw.line([chart_pts_ear[i], chart_pts_ear[i+1]], fill="#00d26a", width=2)
        draw.line([chart_pts_mar[i], chart_pts_mar[i+1]], fill="#ffa502", width=2)
        
    draw.text((380, 550), "-- EAR Stream (Green)", fill="#00d26a", font_size=11)
    draw.text((540, 550), "-- MAR Stream (Orange)", fill="#ffa502", font_size=11)
    draw.line([(380, 595), (1155, 595)], fill="#ff4757", width=1)
    draw.text((1010, 580), "EAR Alarm Threshold (0.22)", fill="#ff4757", font_size=10)

    img.save("report_assets/fig_streamlit_dashboard.png")
    print("fig_streamlit_dashboard.png created successfully (Resolution: 1200x720)")

if __name__ == "__main__":
    generate_hud_quad()
    generate_streamlit_dashboard()
