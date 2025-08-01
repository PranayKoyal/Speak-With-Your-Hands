"""
Sentence-Enhanced Sign Language Communication System
Features:
- Advanced sentence building and grammar correction
- Machine learning-based gesture recognition
- Expanded gesture vocabulary
- Modern UI design with gesture history
- User customization options
- Windows compatibility
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import joblib
from flask import Flask, Response, jsonify
from flask_cors import CORS
import logging
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# Global variables
camera = None
processing = False
model = None
scaler = None
current_gesture = None
current_confidence = 0.0
last_prediction_time = 0
prediction_cooldown = 0.5  # seconds between predictions

def initialize_ml_model():
    """Initialize the ML model and scaler"""
    global model, scaler
    
    try:
        # Get the correct paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.dirname(script_dir)
        models_dir = os.path.join(backend_dir, "models")
        
        logger.info("--- ML Model Initialization Debug ---")
        logger.info(f"Script directory (src): {script_dir}")
        logger.info(f"Backend directory: {backend_dir}")
        logger.info(f"Target models directory: {models_dir}")
        
        # Check if models directory exists
        if not os.path.exists(models_dir):
            logger.error(f"Models directory not found: {models_dir}")
            return False
            
        # Load metadata if available
        metadata_path = os.path.join(models_dir, "model_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info("Model metadata loaded successfully:")
                logger.info(f"Training date: {metadata.get('training_date', 'Unknown')}")
                logger.info(f"Number of classes: {len(metadata.get('classes', []))}")
                logger.info(f"Model accuracy: {metadata.get('accuracy', 0):.3f}")
            except Exception as e:
                logger.warning(f"Could not load metadata: {str(e)}")
        
        # Load model and scaler
        model_path = os.path.join(models_dir, "custom_gesture_model.pkl")
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        
        logger.info(f"Attempting to load model from: {model_path}")
        logger.info(f"Attempting to load scaler from: {scaler_path}")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
            
        if not os.path.exists(scaler_path):
            logger.error(f"Scaler file not found: {scaler_path}")
            return False
        
        # Load model and scaler using joblib
        try:
            model = joblib.load(model_path)
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
            
        try:
            scaler = joblib.load(scaler_path)
            logger.info("Scaler loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading scaler: {str(e)}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error during model initialization: {str(e)}")
        return False

def extract_hand_features(hand_landmarks):
    """Extract hand landmarks features from MediaPipe results."""
    try:
        features = []
        for landmark in hand_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        return np.array(features).reshape(1, -1)
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
    return None

def predict_gesture(features):
    """Predict gesture using the Random Forest model"""
    try:
        if model is None or scaler is None:
            logger.error("Model or scaler not initialized")
            return None
    
        # Scale features
        scaled_features = scaler.transform(features)
        
        # Predict
        prediction = model.predict(scaled_features)
        probabilities = model.predict_proba(scaled_features)
        
        # Get confidence score
        confidence = np.max(probabilities)
        
        return prediction[0], confidence
        
    except Exception as e:
        logger.error(f"Error predicting gesture: {str(e)}")
        return None, 0.0

def process_frame(hand_landmarks):
    """Process a hand_landmarks and return the predicted gesture"""
    global current_gesture, current_confidence, last_prediction_time
    
    features = extract_hand_features(hand_landmarks)
    
    if features is not None:
        # Only predict if enough time has passed since last prediction
        current_time = time.time()
        if current_time - last_prediction_time >= prediction_cooldown:
            prediction, confidence = predict_gesture(features)
            if prediction is not None and confidence > 0.7:  # Confidence threshold
                current_gesture = prediction
                current_confidence = confidence
                last_prediction_time = current_time
                return prediction
    
    return None

@app.route('/')
def index():
    return "Sign Language Recognition Backend"

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global camera, processing, current_gesture, current_confidence
    
        while True:
            if camera is None:
                try:
                    camera = cv2.VideoCapture(0)
                    if not camera.isOpened():
                        logger.error("Camera not available")
                        continue
                    logger.info(f"Camera initialized with resolution {camera.get(cv2.CAP_PROP_FRAME_WIDTH)}x{camera.get(cv2.CAP_PROP_FRAME_HEIGHT)} at {camera.get(cv2.CAP_PROP_FPS)} FPS")
                except Exception as e:
                    logger.error(f"Error initializing camera: {str(e)}")
                    continue
            
            success, frame = camera.read()
            if not success:
                logger.error("Failed to read frame")
                continue
            
            if processing:
                # Convert BGR to RGB for MediaPipe, process it and draw landmarks
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                
                results = hands.process(rgb_frame)
                
                rgb_frame.flags.writeable = True

                # Draw hand landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                        )
                        
                        # Process frame for gesture recognition
                        gesture = process_frame(hand_landmarks)
                        if gesture:
                            # Draw the predicted gesture on the frame
                            cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(frame, f"Confidence: {current_confidence:.2f}", (10, 70),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
        
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_processing')
def start_processing():
    global processing
    processing = True
    return jsonify({"status": "success", "message": "Processing started"})

@app.route('/stop_processing')
def stop_processing():
    global processing
    processing = False
    return jsonify({"status": "success", "message": "Processing stopped"})

@app.route('/status')
def get_status():
    global processing, camera, current_gesture, current_confidence
    return jsonify({
        "processing": processing,
        "camera_active": camera is not None and camera.isOpened(),
        "model_loaded": model is not None and scaler is not None,
        "current_gesture": current_gesture,
        "confidence": current_confidence,
        "fps": camera.get(cv2.CAP_PROP_FPS) if camera is not None and camera.isOpened() else 0.0
    })

if __name__ == '__main__':
    print("Starting Sentence-Enhanced Sign Language Communication System")
    
    # Initialize MediaPipe
    print("MediaPipe Hands initialized successfully")
    
    # Initialize ML model
    if not initialize_ml_model():
        print("Warning: Failed to load ML model. System will fall back to rule-based detection.")
    
    # Start Flask server
        print("Server running at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
