FROM python:3.10-slim

WORKDIR /app

# System dependencies for Mediapipe + WebRTC + OpenCV
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --default-timeout=200 --retries=20 --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
