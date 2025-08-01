import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pickle
import mediapipe as mp
import cv2
from sklearn.model_selection import cross_val_score
import urllib.request
import zipfile
import warnings
warnings.filterwarnings('ignore')

class SignLanguageEvaluator:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        
        # Define the gestures (ASL alphabet)
        self.gestures = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
        
        self.model = None
        self.scaler = None
        self.X = None
        self.y = None
    
    def download_dataset(self):
        """
        Download and prepare the Sign Language dataset
        """
        print("Downloading Sign Language dataset...")
        
        # Download training data
        url = "https://github.com/mon95/Sign-Language-and-Static-gesture-recognition-using-sklearn/raw/master/Dataset.zip"
        zip_path = "sign_language_dataset.zip"
        
        try:
            urllib.request.urlretrieve(url, zip_path)
            
            # Extract dataset
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("dataset")
            
            # Remove zip file
            os.remove(zip_path)
            
            print("Dataset downloaded and extracted successfully!")
            return True
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False
    
    def load_and_preprocess_data(self):
        """
        Load and preprocess the dataset
        """
        print("Loading and preprocessing data...")
        
        try:
            # Load dataset
            data_path = "dataset/Dataset"
            X = []
            y = []
            
            # Iterate through all user directories
            for user_dir in os.listdir(data_path):
                user_path = os.path.join(data_path, user_dir)
                if not os.path.isdir(user_path):
                    continue
                
                # Process all images in the user directory
                for img_name in os.listdir(user_path):
                    if not img_name.endswith(('.jpg', '.jpeg', '.png')):
                        continue
                        
                    # Get the gesture label from the first character of the filename
                    gesture = img_name[0]
                    
                    img_path = os.path.join(user_path, img_name)
                    
                    # Read image
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                    
                    # Convert to RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Process with MediaPipe
                    results = self.hands.process(image_rgb)
                    
                    if results.multi_hand_landmarks:
                        # Extract features from the first hand
                        features = []
                        for landmark in results.multi_hand_landmarks[0].landmark:
                            features.extend([landmark.x, landmark.y, landmark.z])
                        
                        X.append(features)
                        y.append(gesture)
            
            self.X = np.array(X)
            self.y = np.array(y)
            
            print(f"Dataset loaded: {len(self.X)} samples")
            unique_gestures = sorted(list(set(self.y)))
            print(f"Unique gestures found: {unique_gestures}")
            print(f"Number of unique gestures: {len(unique_gestures)}")
            return True
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def train_and_evaluate(self, test_size=0.2):
        """
        Train and evaluate the model
        """
        if self.X is None or self.y is None:
            print("No data available. Please load the dataset first.")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"\nCross-validation scores: {cv_scores}")
        print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        unique_labels = sorted(list(set(y_test)))
        conf_matrix = confusion_matrix(y_test, y_pred, labels=unique_labels)
        class_report = classification_report(y_test, y_pred, labels=unique_labels)
        
        print("\nModel Performance Metrics:")
        print(f"Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(class_report)
        
        # Plot confusion matrix
        plt.figure(figsize=(15, 12))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=unique_labels,
            yticklabels=unique_labels
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Plot feature importance
        self.plot_feature_importance()
        
        return accuracy, conf_matrix, class_report
    
    def plot_feature_importance(self, top_n=20):
        """
        Plot feature importance
        """
        if self.model is None:
            print("Model not trained yet")
            return
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create feature names
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
        Save the trained model and scaler
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