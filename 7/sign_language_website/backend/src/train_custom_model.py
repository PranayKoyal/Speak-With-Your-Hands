"""
Script to train a custom sign language gesture recognition model using user-provided datasets.

Loads images from specified directories (e.g., user_datasets/alphabet_train),
extracts MediaPipe hand landmarks, trains a RandomForestClassifier, and saves
the trained model and scaler.
"""

import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
import argparse
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CustomModelTrainer")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def extract_landmarks(image_path):
    """Extracts MediaPipe hand landmarks from a single image."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Could not read image: {image_path}")
            return None

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Process the image and find hands
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True

        if results.multi_hand_landmarks:
            # Assuming only one hand per image for sign language gestures
            hand_landmarks = results.multi_hand_landmarks[0]
            # Extract landmark coordinates (x, y, z) relative to wrist
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            features = []
            for landmark in hand_landmarks.landmark:
                features.extend([landmark.x - wrist.x, landmark.y - wrist.y, landmark.z - wrist.z])
            return np.array(features)
        else:
            # logger.debug(f"No hand landmarks detected in: {image_path}")
            return None
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return None

def load_data_from_directory(data_dir):
    """Loads image data and extracts landmarks from a directory structure.

    Assumes data_dir contains subdirectories, each named after a gesture class,
    and each subdirectory contains images of that gesture.
    """
    features = []
    labels = []
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    if not class_names:
        logger.error(f"No subdirectories found in {data_dir}. Expecting class folders.")
        return None, None, None

    logger.info(f"Found classes: {class_names}")

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        logger.info(f"Processing class: {class_name} from {class_dir}")
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_files:
            logger.warning(f"No images found in directory: {class_dir}")
            continue

        for image_file in tqdm(image_files, desc=f"Processing {class_name}"):
            image_path = os.path.join(class_dir, image_file)
            landmarks = extract_landmarks(image_path)
            if landmarks is not None:
                features.append(landmarks)
                labels.append(class_name)
            # else: # Optional: Log skipped images
                # logger.debug(f"Skipped image due to no landmarks: {image_path}")

    if not features:
        logger.error("No features extracted. Check dataset structure and image quality.")
        return None, None, None

    return np.array(features), np.array(labels), class_names

def main(args):
    logger.info("Starting custom model training...")

    # --- 1. Load Data --- #
    logger.info(f"Loading data from: {args.data_dir}")
    features, labels, class_names = load_data_from_directory(args.data_dir)

    if features is None or labels is None:
        logger.error("Failed to load data. Exiting.")
        return

    logger.info(f"Loaded {len(features)} samples from {len(class_names)} classes.")
    logger.info(f"Feature shape: {features.shape}, Labels shape: {labels.shape}")

    if len(features) < 2 or len(np.unique(labels)) < 2:
        logger.error("Insufficient data or classes for training. Need at least 2 samples and 2 classes.")
        return

    # --- 2. Preprocess Data --- #
    logger.info("Preprocessing data...")
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Split data into training and testing sets
    test_size = args.test_split
    if len(features) * (1 - test_size) < len(class_names):
        logger.warning(f"Test split {test_size} too large for the number of samples and classes. Adjusting...")
        # Adjust test_size to ensure at least one sample per class in the training set
        min_train_samples = len(class_names)
        max_test_samples = len(features) - min_train_samples
        test_size = min(test_size, max_test_samples / len(features))
        logger.info(f"Adjusted test split to {test_size:.2f}")

    if test_size <= 0 or test_size >= 1:
         logger.warning(f"Cannot perform train/test split with test_size={test_size}. Using all data for training.")
         X_train, X_test, y_train, y_test = features_scaled, np.array([]), labels, np.array([])
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                features_scaled, labels, test_size=test_size, random_state=42, stratify=labels
            )
            logger.info(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")
        except ValueError as e:
            logger.warning(f"Stratified split failed ({e}). Performing non-stratified split.")
            X_train, X_test, y_train, y_test = train_test_split(
                features_scaled, labels, test_size=test_size, random_state=42
            )
            logger.info(f"Data split (non-stratified): {len(X_train)} training samples, {len(X_test)} testing samples.")

    # --- 3. Train Model --- #
    logger.info("Training RandomForestClassifier model...")
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    logger.info("Model training completed.")

    # --- 4. Evaluate Model (Optional) --- #
    if X_test.size > 0:
        logger.info("Evaluating model on test set...")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Test Set Accuracy: {accuracy * 100:.2f}%")
    else:
        logger.info("No test set available for evaluation.")

    # --- 5. Save Model and Scaler --- #
    model_path = os.path.join(args.output_dir, args.model_name + ".pkl")
    scaler_path = os.path.join(args.output_dir, args.scaler_name + ".pkl")

    logger.info(f"Saving model to: {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    logger.info(f"Saving scaler to: {scaler_path}")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    logger.info("Custom model training finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a custom sign language gesture recognition model.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing class subdirectories with gesture images (e.g., ../user_datasets/alphabet_train)")
    parser.add_argument("--output_dir", type=str, default="..",
                        help="Directory to save the trained model and scaler (default: parent directory)")
    parser.add_argument("--model_name", type=str, default="custom_gesture_model",
                        help="Filename for the saved model (without .pkl extension)")
    parser.add_argument("--scaler_name", type=str, default="custom_gesture_scaler",
                        help="Filename for the saved scaler (without .pkl extension)")
    parser.add_argument("--n_estimators", type=int, default=100,
                        help="Number of trees in the RandomForest model")
    parser.add_argument("--test_split", type=float, default=0.2,
                        help="Proportion of data to use for the test set (0.0 to 1.0)")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
