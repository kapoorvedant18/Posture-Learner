FROM python:3.10-slim

WORKDIR /app

# Install basic system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
