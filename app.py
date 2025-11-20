import gradio as gr
import cv2
import mediapipe as mp
import numpy as np
import joblib
import math

# Load model
model = joblib.load("posture_model_xgb.pkl")
encoder = joblib.load("label_encoder.pkl")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

VISIBILITY_THRESHOLD = 0.2
LM_EAR = mp_pose.PoseLandmark.RIGHT_EAR
LM_SHO = mp_pose.PoseLandmark.RIGHT_SHOULDER
LM_HIP = mp_pose.PoseLandmark.RIGHT_HIP

def detect_posture(frame):
    if frame is None:
        return frame, "No frame"

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img)

    if not results.pose_landmarks:
        return frame, "Landmarks not visible. Ensure your full upper body is in frame."

    lm_ear = results.pose_landmarks.landmark[LM_EAR]
    lm_sho = results.pose_landmarks.landmark[LM_SHO]
    lm_hip = results.pose_landmarks.landmark[LM_HIP]

    if (
        lm_ear.visibility < VISIBILITY_THRESHOLD or
        lm_sho.visibility < VISIBILITY_THRESHOLD or
        lm_hip.visibility < VISIBILITY_THRESHOLD
    ):
        return frame, "Your full upper body is not visible. Please move back."

    EAR = np.array([lm_ear.x, lm_ear.y, lm_ear.z])
    SHO = np.array([lm_sho.x, lm_sho.y, lm_sho.z])
    HIP = np.array([lm_hip.x, lm_hip.y, lm_hip.z])

    torso_len = np.linalg.norm(SHO[:2] - HIP[:2])
    if torso_len < 0.0001:
        return frame, "Move back from camera."

    torso_angle = math.degrees(math.atan2(HIP[1] - SHO[1], HIP[0] - SHO[0]))
    ear_x_offset_norm = (EAR[0] - SHO[0]) / torso_len
    ear_y_offset_norm = (EAR[1] - SHO[1]) / torso_len
    shoulder_hip_slope = (SHO[1] - HIP[1]) / (SHO[0] - HIP[0])

    features = np.array([[torso_angle, ear_x_offset_norm, ear_y_offset_norm, shoulder_hip_slope]])
    pred = model.predict(features)[0]
    label = encoder.inverse_transform([pred])[0]

    status = "GOOD POSTURE" if label == "good" else "BAD POSTURE"

    # Draw label on frame
    out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.putText(out, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0, 255, 0) if label == "good" else (0, 0, 255), 3)

    return out, status


with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align:center;'>Posture Learner</h1>")

    webcam = gr.Video(source="webcam", streaming=True)
    output_img = gr.Image()
    output_text = gr.Textbox(label="Output")

    webcam.stream(detect_posture, inputs=webcam, outputs=[output_img, output_text])

demo.launch(server_name="0.0.0.0", server_port=7860)
