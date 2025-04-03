# Use the official lightweight Python image.
FROM python:3.11-slim

# Ensure Python output is logged immediately.
ENV PYTHONUNBUFFERED True

# Set the working directory in the container.
WORKDIR /app

# Copy local code (including shape_predictor_68_face_landmarks.dat) to the container.
COPY . ./

# Install system dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip.
RUN pip install --upgrade pip

# Install all Python dependencies from your requirements.txt.
RUN pip install -r requirements.txt

# Expose the port that the Flask app will run on.
EXPOSE 5000

# Start the Flask app using gunicorn.
# This assumes your Flask app instance is named "app" in main.py.
CMD ["python", "app.py"]
