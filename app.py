import cv2
import mediapipe as mp
import numpy as np
import joblib
import gradio as gr

# Load model + encoder
model = joblib.load("posture_model_xgb.pkl")
encoder = joblib.load("label_encoder.pkl")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

def detect_posture(frame):
    if frame is None:
        return None, "Waiting for camera..."

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return frame, "No landmarks detected. Please move back."

    landmarks = results.pose_landmarks.landmark

    needed = [
        landmarks[mp_pose.PoseLandmark.RIGHT_EAR],
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    ]

    if any(p.visibility < 0.5 for p in needed):
        return frame, "Your full upper body is not visible."

    pts = np.array([[p.x, p.y, p.z] for p in needed]).flatten().reshape(1, -1)
    pred = model.predict(pts)
    label = encoder.inverse_transform(pred)[0]

    # Draw output text on frame
    annotated = frame.copy()
    cv2.putText(annotated, f"Posture: {label}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    return annotated, label


with gr.Blocks() as app:
    gr.Markdown("<h1 style='text-align:center;'>Posture Learner</h1>")

    webcam = gr.Video(source="webcam", label="Sit in front of camera", streaming=True)
    output_img = gr.Image(label="Processed Frame")
    output_text = gr.Textbox(label="Prediction")

    start_btn = gr.Button("Start")
    stop_btn = gr.Button("Stop")

    # Start streaming
    start_btn.click(fn=None, inputs=None, outputs=None,
                    _js="() => { navigator.mediaDevices.getUserMedia({video:true}); }")

    # process each frame
    webcam.change(detect_posture, inputs=webcam, outputs=[output_img, output_text])

    # Stop webcam
    stop_btn.click(fn=None, inputs=None, outputs=None,
                   _js="() => { const v=document.querySelector('video'); v.srcObject.getTracks().forEach(t=>t.stop()); }")

app.queue()
app.launch(server_name="0.0.0.0", server_port=7860)
