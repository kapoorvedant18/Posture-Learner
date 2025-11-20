import gradio as gr
import cv2
import mediapipe as mp
import numpy as np
import joblib
import math

# Load model & encoder
model = joblib.load("posture_model_xgb.pkl")
encoder = joblib.load("label_encoder.pkl")

# Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

VISIBILITY_THRESHOLD = 0.2
LM_EAR = mp_pose.PoseLandmark.RIGHT_EAR
LM_SHO = mp_pose.PoseLandmark.RIGHT_SHOULDER
LM_HIP = mp_pose.PoseLandmark.RIGHT_HIP


def process_frame(frame):
    """
    Receives a webcam frame from Gradio
    Returns:
       processed_frame, posture_label
    """

    if frame is None:
        return None, "No Frame"

    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Run Mediapipe
    results = pose.process(frame)

    if not results.pose_landmarks:
        return frame, "Please enter the frame"

    lm_ear = results.pose_landmarks.landmark[LM_EAR]
    lm_sho = results.pose_landmarks.landmark[LM_SHO]
    lm_hip = results.pose_landmarks.landmark[LM_HIP]

    # If key points not visible
    if (
        lm_ear.visibility < VISIBILITY_THRESHOLD or
        lm_sho.visibility < VISIBILITY_THRESHOLD or
        lm_hip.visibility < VISIBILITY_THRESHOLD
    ):
        return frame, "Move slightly back from camera"

    # Extract features
    EAR = np.array([lm_ear.x, lm_ear.y, lm_ear.z])
    SHO = np.array([lm_sho.x, lm_sho.y, lm_sho.z])
    HIP = np.array([lm_hip.x, lm_hip.y, lm_hip.z])

    torso_len = np.linalg.norm(SHO[:2] - HIP[:2])
    if torso_len < 0.0001:
        return frame, "Move back from camera"

    torso_angle = math.degrees(math.atan2(HIP[1] - SHO[1], HIP[0] - SHO[0]))
    ear_x_offset_norm = (EAR[0] - SHO[0]) / torso_len
    ear_y_offset_norm = (EAR[1] - SHO[1]) / torso_len
    shoulder_hip_slope = (SHO[1] - HIP[1]) / (SHO[0] - HIP[0])

    features = np.array([[torso_angle, ear_x_offset_norm, ear_y_offset_norm, shoulder_hip_slope]])
    pred = model.predict(features)[0]
    label = encoder.inverse_transform([pred])[0]

    # Draw posture label on frame
    color = (0, 255, 0) if label == "good" else (255, 0, 0)
    cv2.putText(img, f"Posture: {label.upper()}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Draw keypoints
    mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    final_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return final_img, label


# Build the Gradio App
with gr.Blocks(title="Posture Learner") as demo:
    gr.Markdown("<h1 style='text-align:center;'>Posture Learner</h1>", unsafe_allow_html=True)

    with gr.Row():
        cam = gr.Image(source="webcam", streaming=True, label="Camera", height=350)
        output_img = gr.Image(label="Processed Feed", height=350)
        status = gr.Textbox(label="Posture Status")

    cam.stream(process_frame, inputs=cam, outputs=[output_img, status])

demo.launch()
