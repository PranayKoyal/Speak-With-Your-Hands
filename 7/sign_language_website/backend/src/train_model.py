import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import mediapipe as mp
import cv2
import time

def create_gesture_dataset():
    """
    Create a dataset of hand gestures using MediaPipe
    """
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return None, None

    # Prepare data collection
    gestures = [
        "OPEN_PALM", "CLOSED_FIST", "THUMBS_UP", "THUMBS_DOWN", "VICTORY",
        "POINTING_UP", "OK_SIGN", "PINCH", "SPREAD_FINGERS", "INDEX_POINT",
        "ROCK_SIGN", "PINKY_UP", "HAND_WAVE", "FIST_BUMP"
    ]
    
    X = []  # Features
    y = []  # Labels
    
    print("Starting data collection...")
    print("Press 'q' to quit data collection")
    
    for gesture in gestures:
        print(f"\nCollecting data for gesture: {gesture}")
        print("Press 's' to start collecting samples (30 samples will be collected)")
        print("Make sure your hand is clearly visible to the camera")
        
        samples_collected = 0
        collecting = False
        
        while samples_collected < 30:  # Collect 30 samples per gesture
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = hands.process(frame_rgb)
            
            # Draw hand landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
                    
                    if collecting:
                        # Extract features (x, y, z coordinates of each landmark)
                        features = []
                        for landmark in hand_landmarks.landmark:
                            features.extend([landmark.x, landmark.y, landmark.z])
                        
                        X.append(features)
                        y.append(gesture)
                        samples_collected += 1
                        print(f"Collected sample {samples_collected}/30 for {gesture}")
                        time.sleep(0.1)  # Small delay between samples
            
            # Display status
            cv2.putText(
                frame,
                f"Gesture: {gesture} - Samples: {samples_collected}/30",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            cv2.imshow('Gesture Collection', frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                collecting = True
                print("Started collecting samples...")
        
        if samples_collected < 30:
            print(f"Warning: Only collected {samples_collected} samples for {gesture}")
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    return np.array(X), np.array(y)

def train_and_save_model(X, y):
    """
    Train the model and save it along with the scaler
    """
    # Create models directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(backend_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Initialize and fit the scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize and train the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42
    )
    model.fit(X_scaled, y)
    
    # Save the model and scaler
    model_path = os.path.join(models_dir, 'gesture_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f, protocol=4)  # Use protocol 4 for better compatibility
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f, protocol=4)  # Use protocol 4 for better compatibility
    
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    
    return model, scaler

def main():
    print("Starting gesture recognition model training...")
    
    # Create dataset
    X, y = create_gesture_dataset()
    if X is None or y is None:
        print("Error: Failed to create dataset")
        return
    
    print(f"\nDataset created successfully:")
    print(f"Number of samples: {len(X)}")
    print(f"Number of features per sample: {X.shape[1]}")
    print(f"Unique gestures: {len(set(y))}")
    
    # Train and save model
    print("\nTraining model...")
    model, scaler = train_and_save_model(X, y)
    
    print("\nTraining completed successfully!")
    print("You can now use the model for gesture recognition.")

if __name__ == "__main__":
    main() 