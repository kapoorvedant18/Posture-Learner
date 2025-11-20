import cv2
import mediapipe as mp
import numpy as np
import joblib
import gradio as gr

# Load model + label encoder
model = joblib.load("posture_model_xgb.pkl")
encoder = joblib.load("label_encoder.pkl")

# Mediapipe pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Extract required posture landmarks
REQUIRED = {
    "RIGHT_EAR": mp_pose.PoseLandmark.RIGHT_EAR,
    "RIGHT_SHOULDER": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "RIGHT_HIP": mp_pose.PoseLandmark.RIGHT_HIP
}

def extract_landmarks(image):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        return None, "No person detected. Please enter the frame."

    lm = results.pose_landmarks.landmark

    points = []
    for name, idx in REQUIRED.items():
        if lm[idx].visibility < 0.5:
            return None, "Your full upper body is not visible. Please move back."
        points.extend([lm[idx].x, lm[idx].y, lm[idx].z])

    return np.array(points).reshape(1, -1), None


def classify_posture(image):
    landmarks, error = extract_landmarks(image)

    if error:
        return error

    prediction = model.predict(landmarks)
    label = encoder.inverse_transform(prediction)[0]

    return f"Posture: {label}"


# Gradio UI
interface = gr.Interface(
    fn=classify_posture,
    inputs=gr.Image(source="webcam", streaming=True),
    outputs="text",
    live=True,
    title="Posture Learner",
    description="Sit in front of the camera. The model will classify your posture in real time."
)


# 🚀 IMPORTANT FOR RAILWAY DEPLOYMENT
if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

