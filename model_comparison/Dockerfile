
# Use official Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Install dependencies
RUN pip install ultralytics opencv-python pandas matplotlib

# Copy model and source files
COPY images /app/images
COPY script.py /app/script.py

# Run inference
CMD ["python", "script.py"]
