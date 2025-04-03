import os
import cv2
import dlib
import numpy as np
from imutils import face_utils
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
import base64

# Define paths relative to the current file location.
script_dir = os.path.dirname(__file__)
predictor_path = os.path.join(script_dir, "shape_predictor_68_face_landmarks.dat")
model_path_gaze = os.path.join(script_dir, "models", "gazev3.1.h5")

# Load face detector, landmark predictor, and the pretrained gaze model.
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
model_gaze = load_model(model_path_gaze)

# Define image size and class labels for the gaze model.
IMG_SIZE = (64, 56)  # (width, height)
class_labels = ['center', 'left', 'right']

def detect_gaze(eye_img):
    """Runs the loaded gaze model on a preprocessed eye image."""
    preds = model_gaze.predict(eye_img)
    gaze_idx = int(np.argmax(preds[0]))
    return class_labels[gaze_idx]

def crop_eye(gray, eye_points):
    """
    Crops the eye region from a grayscale face image, 
    given an array of landmark points (6 points for one eye).
    """
    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = (x2 - x1) * 1.2
    h = w * IMG_SIZE[1] / IMG_SIZE[0]
    margin_x, margin_y = w / 2, h / 2

    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)
    eye_img = gray[min_y:max_y, min_x:max_x]
    return eye_img

def process_frame_cv(frame):
    """
    1) Converts the frame to grayscale.
    2) Detects the first face with dlib.
    3) Extracts the left eye region (landmarks 36 to 41).
    4) Resizes and normalizes the eye image for the gaze model.
    5) Returns the detected gaze (string) or an error message.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return "No face detected"

    face = faces[0]
    shape = predictor(gray, face)
    shape_np = face_utils.shape_to_np(shape)

    # Left eye landmarks: 36 to 41
    eye_img_l = crop_eye(gray, shape_np[36:42])
    if eye_img_l.size == 0:
        return "Could not crop left eye"

    # Resize to (width=64, height=56) and normalize
    eye_img_l_resized = cv2.resize(eye_img_l, IMG_SIZE)
    eye_input = eye_img_l_resized.reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.0

    gaze = detect_gaze(eye_input)
    return f"Gaze: {gaze}"

def process_straight_frame(image_bytes):
    """
    Accepts JPEG-encoded image bytes, decodes them into an OpenCV BGR frame,
    then calls process_frame_cv() to perform gaze detection.
    """
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return "Error: Could not decode image"
    return process_frame_cv(frame)

# Create the Flask application.
app = Flask(__name__)

@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    data = request.get_json()
    image_data = data.get('image_data')
    if not image_data:
        return jsonify({'error': 'No image data provided'}), 400

    # Decode the base64-encoded image.
    try:
        image_bytes = base64.b64decode(image_data)
    except Exception as e:
        return jsonify({'error': f'Invalid image data: {e}'}), 400

    # Process the image using the gaze detection pipeline.
    result = process_straight_frame(image_bytes)
    return jsonify({'result': result})

if __name__ == '__main__':
    # Run the Flask app on port 5000, accessible from all interfaces.
    app.run(host='0.0.0.0', port=5000)
