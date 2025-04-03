# Use an unpinned slim version so you always get the latest updates.
FROM python:3.11

# Install system dependencies needed to build dlib and run OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgtk-3-dev \
    libboost-all-dev \
    python3-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version.
RUN pip install --no-cache-dir --upgrade pip

# Create and set the working directory.
WORKDIR /app

# Copy the requirements.txt file.
COPY requirements.txt /app

# Install the Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code and models.
COPY . /app

# Expose the port the Flask app will run on.
EXPOSE 5000

# Start the Flask app.
CMD ["python", "app.py"]
