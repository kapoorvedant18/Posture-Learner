FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
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

# Copy everything
COPY . .

# Install mediapipe first if using local wheel
RUN pip install --no-cache-dir ./mediapipe-0.10.7-cp310-cp310-manylinux_2_17_x86_64.whl

# Install other python dependencies
RUN pip install --default-timeout=200 --retries=20 --no-cache-dir -r requirements.txt

# Start Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
