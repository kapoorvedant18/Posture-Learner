FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for ffmpeg, OpenCV, and PyAV
RUN apt-get update && apt-get install -y \
    ffmpeg \
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libsm6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --default-timeout=200 --retries=20 --no-cache-dir -r requirements.txt

# Start Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
