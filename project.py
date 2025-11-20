import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import numpy as np
import joblib
import math

st.set_page_config(page_title="Posture Learner", layout="wide")

# ----------- GLOBAL CSS (text centering + UI cleanup) -----------
st.markdown("""
    <style>
        /* Center all main content */
        h1, h2, h3, h4, h5, h6, p, div, span {
            text-align: center !important;
        }

        /* Sidebar left-aligned */
        section[data-testid="stSidebar"] * {
            text-align: left !important;
        }

        /* Hide WebRTC controls */
        video::-webkit-media-controls {
            display:none !important;
        }
    </style>
""", unsafe_allow_html=True)


# ------------------- SIDEBAR -------------------
st.sidebar.title("Posture Learner")
st.sidebar.markdown("A real-time posture detection system built using:")
st.sidebar.markdown("- Mediapipe Pose")
st.sidebar.markdown("- XGBoost Classifier")
st.sidebar.markdown("- Streamlit + WebRTC")

st.sidebar.markdown("---")
st.sidebar.markdown("### How to Use")
st.sidebar.write("""
1. Sit in front of the camera.  
2. Ensure your upper body is visible.  
3. Maintain good lighting.  
4. System will automatically show GOOD or BAD posture.  
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Info")
st.sidebar.write("• 4 engineered features\n• Predicts: **GOOD** or **BAD** posture")


# ------------------- PAGE HEADER -------------------
st.markdown("<h1>Posture Learner – Real-Time Detection</h1>", unsafe_allow_html=True)
st.markdown("<p></p>", unsafe_allow_html=True)


# ------------------- LOAD MODEL -------------------
model = joblib.load("posture_model_xgb.pkl")
encoder = joblib.load("label_encoder.pkl")

# Mediapipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

VIS_THRESHOLD = 0.3
LM_EAR = mp_pose.PoseLandmark.RIGHT_EAR
LM_SHO = mp_pose.PoseLandmark.RIGHT_SHOULDER
LM_HIP = mp_pose.PoseLandmark.RIGHT_HIP


# ------------------- VIDEO TRANSFORMER -------------------
class PostureDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        # CASE 1 → NO LANDMARKS AT ALL
        if not results.pose_landmarks:
            cv2.putText(img, "Please enter the frame",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, (0, 0, 255), 3)
            return cv2.resize(img, (640, 360))

        # Draw pose for user reference
        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        # Extract 3 key landmarks
        lm_ear = results.pose_landmarks.landmark[LM_EAR]
        lm_sho = results.pose_landmarks.landmark[LM_SHO]
        lm_hip = results.pose_landmarks.landmark[LM_HIP]

        # CASE 2 → PARTIAL VISIBILITY (too close, or cropped)
        if (
            lm_ear.visibility < VIS_THRESHOLD or
            lm_sho.visibility < VIS_THRESHOLD or
            lm_hip.visibility < VIS_THRESHOLD
        ):
            cv2.putText(img, "Move away from the camera",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, (255, 255, 0), 3)
            return cv2.resize(img, (640, 360))

        # Now safe to calculate features
        EAR = np.array([lm_ear.x, lm_ear.y, lm_ear.z])
        SHO = np.array([lm_sho.x, lm_sho.y, lm_sho.z])
        HIP = np.array([lm_hip.x, lm_hip.y, lm_hip.z])

        torso_len = np.linalg.norm(SHO[:2] - HIP[:2])

        if torso_len < 0.0001:
            cv2.putText(img, "Move away from the camera",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, (255, 255, 0), 3)
            return cv2.resize(img, (640, 360))

        # Feature extraction
        torso_angle = math.degrees(math.atan2(HIP[1] - SHO[1], HIP[0] - SHO[0]))
        ear_x = (EAR[0] - SHO[0]) / torso_len
        ear_y = (EAR[1] - SHO[1]) / torso_len
        slope = (SHO[1] - HIP[1]) / (SHO[0] - HIP[0])

        features = np.array([[torso_angle, ear_x, ear_y, slope]])
        pred = model.predict(features)[0]
        label = encoder.inverse_transform([pred])[0]

        # Posture label
        if label == "good":
            cv2.putText(img, "GOOD POSTURE", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        else:
            cv2.putText(img, "BAD POSTURE", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        return cv2.resize(img, (640, 360))


# ------------------- CAMERA FEED -------------------
st.markdown("<h3>Camera Feed</h3>", unsafe_allow_html=True)

webrtc_streamer(
    key="posture-basic",
    video_transformer_factory=PostureDetector,
    media_stream_constraints={
        "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
        "audio": False
    },
    async_processing=True,
)
