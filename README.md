# 🚗 Real-Time Driver Drowsiness Detection System

An AI-powered computer vision system designed to monitor driver alertness in real-time, instantly detecting eye closure (microsleep), yawning, and head nodding to trigger life-saving audio-visual alarms.

---

## 🔍 Why the Original Code Failed & What Was Fixed

1. **Failure to Detect Closed Eyes (Haar Cascade Flaw)**:
   - OpenCV Haar Cascades (`haarcascade_eye_tree_eyeglasses.xml`) detect pupils and irises. When eyes close, Haar cascade returns `len(eyes) == 0`.
   - The original code only ran CNN classification `if len(eyes) > 0`. Consequently, **when eyes closed, prediction stopped entirely** and never registered "Eyes Closed".
   - **Fix**: Upgraded to **MediaPipe FaceMesh (478 3D landmarks)** and **Eye Aspect Ratio (EAR)**. Landmark coordinates never vanish when eyes close. Added an anatomical fallback region to Haar mode.

2. **Severe Lag & The Reset Bug ("Not Detecting on Time")**:
   - The original system throttled prediction to 0.5s intervals, required 3/5 votes in a deque (1.5s delay), and then required another 2.0 continuous seconds in `AlertSystem`.
   - Any single frame glitch, face tilt, or lighting shift immediately wiped `closed_start = None`, restarting the entire countdown.
   - **Fix**: Continuous frame-by-frame EAR computation with a temporal tolerance buffer (0.3s). As soon as continuous closure crosses the threshold (e.g., 1.5s), the alarm sounds **immediately on time** without false resets.

3. **Critical Double Normalization Bug**:
   - `eye_model.keras` contains an internal `Rescaling(scale=1/255.0)` layer.
   - The original `eye_detection.py` explicitly divided pixel values by `255.0` before passing them to the model, causing the model to divide by 255 **twice** (`pixel / 65025.0`), destroying CNN accuracy.
   - **Fix**: Corrected data normalization to match model layer architecture.

4. **Silent Audio Failure (Misspelled Path & Thread Crash)**:
   - The directory was named `assests` while code requested `../assets/alarm.mp3`, causing a `FileNotFoundError` that silently set `self.alert = None`.
   - Running Pygame mixer inside a Streamlit WebRTC thread caused audio device lockups.
   - **Fix**: Created multi-path auto-resolution for audio assets, non-blocking asynchronous audio playback, and a native Windows `winsound.Beep` fallback.

5. **Ineffective Yawn Detection**:
   - The original code used `haarcascade_smile.xml` (looking for smiles!) and simply displayed "Mouth detected".
   - **Fix**: Implemented genuine **Mouth Aspect Ratio (MAR)** measuring vertical lip opening vs. mouth width, detecting prolonged yawns ($MAR > 0.60$) and tracking cumulative yawn frequency.

6. **Added Head Nodding / Microsleep Slump Detection**:
   - Uses 3D facial landmarks with `cv2.solvePnP` to estimate head pitch and roll. If the driver's head nods forward or slumps, a warning triggers immediately.

---

## 🚀 How to Run

### Option 1: Native High-Performance Desktop App (Recommended for 30-60+ FPS)
Run the desktop OpenCV app with instant sound alarms and real-time HUD:

```bash
python run_detector.py
```

**Desktop Keyboard Controls:**
- `Q` or `ESC` : Quit
- `M` : Cycle Detection Mode (`Hybrid` ➔ `FaceMesh` ➔ `CNN` ➔ `Haar`)
- `A` : Toggle Audio Alarm (Mute / Unmute)
- `+` / `-` : Adjust EAR Eye Closure sensitivity threshold
- `R` : Reset Counters (Blinks, Yawns, Fatigue Score)

---

### Option 2: Streamlit Interactive Web Dashboard
Run the modern browser dashboard:

```bash
streamlit run app.py
```

**Features in Streamlit:**
- **Live Native Camera Feed**: Ultra-fast camera stream directly inside the browser with live HUD and metrics cards.
- **Browser WebRTC Camera**: WebRTC browser streaming option.
- **Video File Upload**: Upload and analyze driving video files (`.mp4`, `.avi`, `.mov`).
- **Sidebar Sliders**: Live adjustments for Alarm Duration Threshold, EAR sensitivity, MAR yawn threshold, and backend selection.

---

## 📁 Project Structure

```text
minipro/
├── assets/
│   └── alarm.mp3                     # Alarm audio file
├── models/
│   ├── eye_model.keras               # Trained eye closure CNN
│   ├── drowsiness_model.keras        # Multi-class drowsiness CNN
│   └── face_landmarker.task          # MediaPipe 3D FaceLandmarker model
├── modules/
│   ├── __init__.py                   # Package exports
│   ├── alert_system.py               # Robust non-blocking alarm engine
│   ├── eye_detection.py              # EAR calculation & CNN eye inference
│   ├── yawning_detection.py          # MAR calculation & yawn detection
│   └── drowsiness_detector.py        # Master multi-modal detection engine
├── tests/
│   └── test_drowsiness.py            # Unit test suite
├── app.py                            # Streamlit Web Application
├── run_detector.py                   # High-FPS Native Desktop Application
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation
```

---

## 🧪 Running the Tests

To verify all components, run:

```bash
python -m unittest tests/test_drowsiness.py
```
All unit tests verify audio path resolution, EAR/MAR calculations, state machine timing, and frame processing.
