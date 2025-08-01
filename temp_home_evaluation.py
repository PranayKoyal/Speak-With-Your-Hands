import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import mediapipe as mp
import cv2
from sklearn.model_selection import cross_val_score
import os
import warnings
import joblib
warnings.filterwarnings('ignore')

class HomeDatasetEvaluator:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.3,  # Lowered from 0.7 to be more lenient
            min_tracking_confidence=0.3
        )
        
        self.model = None
        self.scaler = None
        self.X = None
        self.y = None
    
    def preprocess_image(self, image):
        """
        Preprocess image to improve hand detection
        """
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Resize to a standard size
        image = cv2.resize(image, (224, 224))
        
        # Enhance contrast
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def load_and_preprocess_data(self, data_path):    
        """
        Load and preprocess the dataset from user directories
        """
        print("Loading and preprocessing your dataset...")
        
        try:
            X = []
            y = []
            stats = {}
            total_processed = 0
            total_skipped = 0
            
            # Process all user directories
            for user_dir in sorted(os.listdir(data_path)):
                user_path = os.path.join(data_path, user_dir)
                if not os.path.isdir(user_path):
                    continue
                    
                print(f"\nProcessing user directory: {user_dir}")
                
                # Process all images in the user directory
                image_files = []
                for root, _, files in os.walk(user_path):
                    for file in files:
                        if file.endswith(('.jpg', '.jpeg', '.png')):
                            image_files.append(os.path.join(root, file))
                
                # Get the gesture label from the parent directory name
                for img_path in image_files:
                    try:
                        # Read image
                        image = cv2.imread(img_path)
                        if image is None:
                            total_skipped += 1
                            continue
                        
                        # Get gesture label from parent directory name
                        gesture = os.path.basename(os.path.dirname(img_path))
                        
                        # Preprocess image
                        processed_image = self.preprocess_image(image)
                        
                        # Process with MediaPipe
                        results = self.hands.process(processed_image)
                        
                        if results.multi_hand_landmarks:
                            # Extract features from the first hand
                            features = []
                            for landmark in results.multi_hand_landmarks[0].landmark:
                                features.extend([landmark.x, landmark.y, landmark.z])
                            
                            X.append(features)
                            y.append(gesture)
                            total_processed += 1
                            
                            # Update stats
                            if gesture not in stats:
                                stats[gesture] = {'processed': 0, 'skipped': 0}
                            stats[gesture]['processed'] += 1
                        else:
                            total_skipped += 1
                            if gesture not in stats:
                                stats[gesture] = {'processed': 0, 'skipped': 0}
                            stats[gesture]['skipped'] += 1
                            
                    except Exception as e:
                        print(f"Error processing image {img_path}: {str(e)}")
                        total_skipped += 1
                        continue
            
            if len(X) == 0:
                print("\nNo valid samples found! Please check the images and path.")
                return False
                
            self.X = np.array(X)
            self.y = np.array(y)
            
            print(f"\nDataset loaded successfully:")
            print(f"Total processed: {total_processed} images")
            print(f"Total skipped: {total_skipped} images")
            print(f"Total samples: {len(self.X)}")
            print(f"Unique gestures found: {sorted(list(set(self.y)))}")
            print(f"Number of unique gestures: {len(set(self.y))}")
            
            print("\nSamples per gesture:")
            for gesture, counts in stats.items():
                print(f"Gesture {gesture}: {counts['processed']} samples")
            
            return True
            
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return False
    
    def evaluate(self, test_size=0.2):
        """
        Evaluate the model and save both user and A-Z datasets
        """
        try:
            if self.X is None or self.y is None:
                print("No data available. Please load the dataset first.")
                return None, None, None
            
            if len(self.X) == 0:
                print("No valid samples found in the dataset.")
                return None, None, None
                
            print(f"\nStarting model evaluation with {len(self.X)} samples...")
            print(f"Number of features: {self.X.shape[1]}")
            print(f"Number of classes: {len(set(self.y))}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            print("\nTraining Random Forest model...")
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Save models and metadata
            print("\nSaving models and metadata...")
            models_dir = os.path.join("7", "sign_language_website", "backend", "models")
            os.makedirs(models_dir, exist_ok=True)
            
            # Save the main model and scaler
            import joblib
            model_path = os.path.join(models_dir, "custom_gesture_model.pkl")
            scaler_path = os.path.join(models_dir, "scaler.pkl")
            
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            
            # Save metadata about the model
            metadata = {
                "features": self.X.shape[1],
                "classes": sorted(list(set(self.y))),
                "num_samples": len(self.X),
                "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "accuracy": accuracy_score(y_test, self.model.predict(X_test_scaled)),
                "cross_val_scores": list(cross_val_score(self.model, X_train_scaled, y_train, cv=5))
            }
            
            metadata_path = os.path.join(models_dir, "model_metadata.json")
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            print(f"Models and metadata saved to {models_dir}")
            
            # Cross-validation
            print("\nPerforming cross-validation...")
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
            print(f"Cross-validation scores: {cv_scores}")
            print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Evaluate on test set
            print("\nEvaluating on test set...")
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
            
            return accuracy, conf_matrix, class_report
            
        except Exception as e:
            print(f"Error during model evaluation: {str(e)}")
            return None, None, None
    
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

    def get_train_test_split(self, test_size=0.2):
        """
        Get train-test split for external visualization
        """
        if self.X is None or self.y is None:
            print("No data available. Please load the dataset first.")
            return None, None, None, None
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
        )
        
        # Scale features
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test

def main():
    # Initialize evaluator
    evaluator = HomeDatasetEvaluator()
    
    # Set the path to your dataset
    data_path = r"C:\Users\bootl\Downloads\7\7\sign_language_website\user_datasets\alphabet_train"
    
    # Load and preprocess data
    if evaluator.load_and_preprocess_data(data_path):
        # Evaluate model
        evaluator.evaluate()

if __name__ == "__main__":
    main() 