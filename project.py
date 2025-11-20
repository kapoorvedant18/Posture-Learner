import gradio as gr
import cv2
import mediapipe as mp
import numpy as np
import joblib
import math

# Load model and encoder
model = joblib.load("posture_model_xgb.pkl")
encoder = joblib.load("label_encoder.pkl")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

VISIBILITY_THRESHOLD = 0.2
LM_EAR = mp_pose.PoseLandmark.RIGHT_EAR
LM_SHO = mp_pose.PoseLandmark.RIGHT_SHOULDER
LM_HIP = mp_pose.PoseLandmark.RIGHT_HIP

def process_frame(frame):
    if frame is None:
        return None, "No frame received"

    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = pose.process(frame)

    if not results.pose_landmarks:
        return img, "Please enter the frame"

    lm_ear = results.pose_landmarks.landmark[LM_EAR]
    lm_sho = results.pose_landmarks.landmark[LM_SHO]
    lm_hip = results.pose_landmarks.landmark[LM_HIP]

    # Check partial visibility
    if (
        lm_ear.visibility < VISIBILITY_THRESHOLD or
        lm_sho.visibility < VISIBILITY_THRESHOLD or
        lm_hip.visibility < VISIBILITY_THRESHOLD
    ):
        return img, "Move back. Some body parts are not visible."

    EAR = np.array([lm_ear.x, lm_ear.y, lm_ear.z])
    SHO = np.array([lm_sho.x, lm_sho.y, lm_sho.z])
    HIP = np.array([lm_hip.x, lm_hip.y, lm_hip.z])

    torso_len = np.linalg.norm(SHO[:2] - HIP[:2])
    if torso_len < 0.0001:
        return img, "Move back from camera"

    torso_angle = math.degrees(math.atan2(HIP[1] - SHO[1], HIP[0] - SHO[0]))
    ear_x_offset_norm = (EAR[0] - SHO[0]) / torso_len
    ear_y_offset_norm = (EAR[1] - SHO[1]) / torso_len
    shoulder_hip_slope = (SHO[1] - HIP[1]) / (SHO[0] - HIP[0])

    features = np.array([[torso_angle, ear_x_offset_norm, ear_y_offset_norm, shoulder_hip_slope]])
    pred = model.predict(features)[0]
    label = encoder.inverse_transform([pred])[0]

    if label == "good":
        text = "GOOD POSTURE"
        color = (0, 255, 0)
    else:
        text = "BAD POSTURE"
        color = (0, 0, 255)

    cv2.putText(img, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    return img, text

# Build Gradio UI
interface = gr.Interface(
    fn=process_frame,
    inputs=gr.Image(source="webcam", streaming=True),
    outputs=[
        gr.Image(label="Camera Output"),
        gr.Textbox(label="Posture Status")
    ],
    title="Posture Learner – Real-Time Detection",
    description="Tracks posture using your webcam and gives feedback instantly."
)

interface.launch()
