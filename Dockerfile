FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 7860

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "app.py"]
