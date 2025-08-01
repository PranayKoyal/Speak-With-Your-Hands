import os
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pickle
import time
from IPython.display import clear_output
import pandas as pd

class GestureRecognitionEvaluator:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Gesture vocabulary
        self.gestures = [
            "OPEN_PALM", "CLOSED_FIST", "THUMBS_UP", "THUMBS_DOWN", "VICTORY",
            "POINTING_UP", "OK_SIGN", "PINCH", "SPREAD_FINGERS", "INDEX_POINT",
            "ROCK_SIGN", "PINKY_UP", "HAND_WAVE", "FIST_BUMP"
        ]
        
        # Initialize data storage
        self.X = []  # Features
        self.y = []  # Labels
        self.model = None
        self.scaler = None
        
        # Performance metrics
        self.training_history = {
            'accuracy': [],
            'val_accuracy': [],
            'samples_per_gesture': {}
        }
    
    def collect_data(self, samples_per_gesture=30):
        """
        Collect training data using webcam
        """
        print("Starting data collection...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return False
        
        try:
            for gesture in self.gestures:
                print(f"\nCollecting data for gesture: {gesture}")
                print("Press 's' to start collecting samples")
                print(f"Target: {samples_per_gesture} samples")
                
                samples_collected = 0
                collecting = False
                
                while samples_collected < samples_per_gesture:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    # Convert to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process with MediaPipe
                    results = self.hands.process(frame_rgb)
                    
                    # Draw hand landmarks if detected
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp.solutions.drawing_utils.draw_landmarks(
                                frame,
                                hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS
                            )
                            
                            if collecting:
                                # Extract features
                                features = []
                                for landmark in hand_landmarks.landmark:
                                    features.extend([landmark.x, landmark.y, landmark.z])
                                
                                self.X.append(features)
                                self.y.append(gesture)
                                samples_collected += 1
                                print(f"Collected sample {samples_collected}/{samples_per_gesture}")
                                time.sleep(0.1)
                    
                    # Display status
                    cv2.putText(
                        frame,
                        f"Gesture: {gesture} - Samples: {samples_collected}/{samples_per_gesture}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    
                    cv2.imshow('Data Collection', frame)
                    
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        collecting = True
                        print("Started collecting samples...")
                
                self.training_history['samples_per_gesture'][gesture] = samples_collected
                
                if samples_collected < samples_per_gesture:
                    print(f"Warning: Only collected {samples_collected} samples for {gesture}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        return True
    
    def prepare_data(self, test_size=0.2):
        """
        Prepare data for training
        """
        X = np.array(self.X)
        y = np.array(self.y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """
        Train the RandomForest model
        """
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance
        """
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred, labels=self.gestures)
        class_report = classification_report(y_test, y_pred, labels=self.gestures)
        
        return accuracy, conf_matrix, class_report
    
    def plot_confusion_matrix(self, conf_matrix):
        """
        Plot confusion matrix heatmap
        """
        plt.figure(figsize=(15, 12))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.gestures,
            yticklabels=self.gestures
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_samples_distribution(self):
        """
        Plot distribution of samples across gestures
        """
        plt.figure(figsize=(15, 6))
        samples = self.training_history['samples_per_gesture']
        plt.bar(samples.keys(), samples.values())
        plt.title('Number of Samples per Gesture')
        plt.xlabel('Gesture')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, top_n=20):
        """
        Plot feature importance
        """
        if self.model is None:
            print("Model not trained yet")
            return
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create feature names (x, y, z for each landmark)
        feature_names = []
        for i in range(21):  # 21 landmarks
            feature_names.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        # Sort and get top N
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(12, 6))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title(f'Top {top_n} Most Important Features')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model_path, scaler_path):
        """
        Save model and scaler
        """
        if self.model is None or self.scaler is None:
            print("Model or scaler not initialized")
            return False
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f, protocol=4)
            
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f, protocol=4)
            
            print(f"Model saved to: {model_path}")
            print(f"Scaler saved to: {scaler_path}")
            return True
        
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, model_path, scaler_path):
        """
        Load saved model and scaler
        """
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            print("Model and scaler loaded successfully")
            return True
        
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def real_time_evaluation(self, duration=60):
        """
        Evaluate model in real-time using webcam
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        start_time = time.time()
        predictions = []
        
        try:
            while (time.time() - start_time) < duration:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = self.hands.process(frame_rgb)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS
                        )
                        
                        # Extract features
                        features = []
                        for landmark in hand_landmarks.landmark:
                            features.extend([landmark.x, landmark.y, landmark.z])
                        
                        # Predict
                        features_scaled = self.scaler.transform([features])
                        prediction = self.model.predict(features_scaled)[0]
                        confidence = np.max(self.model.predict_proba(features_scaled))
                        
                        predictions.append(prediction)
                        
                        # Display prediction
                        cv2.putText(
                            frame,
                            f"{prediction} ({confidence:.2f})",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )
                
                cv2.imshow('Real-time Evaluation', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        # Plot prediction distribution
        if predictions:
            plt.figure(figsize=(12, 6))
            pd.Series(predictions).value_counts().plot(kind='bar')
            plt.title('Distribution of Predictions in Real-time')
            plt.xlabel('Gesture')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

# Example usage in Jupyter Notebook:
"""
# Initialize evaluator
evaluator = GestureRecognitionEvaluator()

# Collect data
evaluator.collect_data(samples_per_gesture=30)

# Prepare data
X_train, X_test, y_train, y_test = evaluator.prepare_data(test_size=0.2)

# Train model
evaluator.train_model(X_train, y_train)

# Evaluate model
accuracy, conf_matrix, class_report = evaluator.evaluate_model(X_test, y_test)

# Print results
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(class_report)

# Plot results
evaluator.plot_confusion_matrix(conf_matrix)
evaluator.plot_samples_distribution()
evaluator.plot_feature_importance()

# Save model
evaluator.save_model('gesture_model.pkl', 'scaler.pkl')

# Real-time evaluation
evaluator.real_time_evaluation(duration=60)  # 60 seconds of real-time evaluation
""" 