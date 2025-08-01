import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score
from temp_home_evaluation import HomeDatasetEvaluator
import os

def calculate_metrics(y_true, y_pred, target_class):
    """Calculate precision, recall, and F1 score for a specific class"""
    y_true_binary = (y_true == target_class).astype(int)
    y_pred_binary = (y_pred == target_class).astype(int)
    
    precision = precision_score(y_true_binary, y_pred_binary)
    recall = recall_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)
    
    return precision, recall, f1

def plot_metrics(evaluator, y_test, y_pred, y_score, class_names):
    """Plot various metrics for model evaluation"""
    try:
        # Create figure with multiple subplots
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(20, 10))
        
        # 1. Bar plot of Accuracy per class
        plt.subplot(2, 2, 1)
        class_accuracy = {}
        for cls in class_names:
            mask = y_test == cls
            class_accuracy[cls] = np.mean(y_pred[mask] == y_test[mask])
        
        accuracy_df = pd.DataFrame(list(class_accuracy.items()), columns=['Class', 'Accuracy'])
        sns.barplot(x='Class', y='Accuracy', data=accuracy_df)
        plt.title('Accuracy per Gesture Class')
        plt.xticks(rotation=45)
        
        # 2. Precision-Recall curve for 'A'
        plt.subplot(2, 2, 2)
        target_class = class_names[0]  # Use first class instead of hardcoding 'A'
        y_true_binary = (y_test == target_class).astype(int)
        class_idx = list(class_names).index(target_class)
        precision, recall, _ = precision_recall_curve(y_true_binary, y_score[:, class_idx])
        avg_precision = average_precision_score(y_true_binary, y_score[:, class_idx])
        
        plt.plot(recall, precision, lw=2, label=f'{target_class} (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for Gesture {target_class}')
        plt.legend(loc='best')
        
        # 3. Metrics Table as a heatmap
        plt.subplot(2, 2, 3)
        precision, recall, f1 = calculate_metrics(y_test, y_pred, target_class)
        
        metrics_dict = {
            'Accuracy': [class_accuracy[target_class]],
            'Precision': [precision],
            'Recall': [recall],
            'F1-Score': [f1]
        }
        metrics_df = pd.DataFrame(metrics_dict, index=[target_class])
        sns.heatmap(metrics_df, annot=True, cmap='YlGnBu', fmt='.3f')
        plt.title(f'Performance Metrics for Gesture {target_class}')
        
        # 4. Feature Importance
        plt.subplot(2, 2, 4)
        evaluator.plot_feature_importance(top_n=10)
        
        plt.tight_layout()
        plt.show()
        
        # Save metrics to CSV
        metrics_df.to_csv('gesture_metrics.csv')
        print("\nMetrics have been saved to 'gesture_metrics.csv'")
        
    except Exception as e:
        print(f"Error in plotting metrics: {str(e)}")
        raise e

def find_dataset_path():
    """Find the correct dataset path"""
    possible_paths = [
        os.path.join("dataset", "Dataset"),
        os.path.join("dataset"),
        os.path.join("..", "dataset"),
        os.path.join("sign_language_website", "user_datasets", "alphabet_train"),
        os.path.join("..", "sign_language_website", "user_datasets", "alphabet_train"),
        os.path.join("user_datasets", "alphabet_train")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found dataset at: {os.path.abspath(path)}")
            return path
            
    print("Could not find dataset directory. Available directories:")
    for path in os.listdir('.'):
        if os.path.isdir(path):
            print(f"- {path}")
    return None

def main():
    try:
        # Initialize evaluator
        evaluator = HomeDatasetEvaluator()
        
        # Find the dataset path
        base_path = find_dataset_path()
        if base_path is None:
            print("Could not find the dataset directory. Please ensure it exists.")
            return
            
        print(f"\nProcessing dataset from: {os.path.abspath(base_path)}")
        
        # Load and preprocess data
        if not evaluator.load_and_preprocess_data(base_path):
            print("Failed to load and preprocess data. Exiting...")
            return
        
        # Train the model and get evaluation results
        print("\nTraining and evaluating the model...")
        accuracy, conf_matrix, class_report = evaluator.evaluate()
        
        if accuracy is None:
            print("Model evaluation failed. Exiting...")
            return
            
        # Get predictions for visualization
        print("\nPreparing visualization data...")
        X_train_scaled, X_test_scaled, y_train, y_test = evaluator.get_train_test_split()
        
        if evaluator.model is None:
            print("Model not properly trained. Exiting...")
            return
            
        y_pred = evaluator.model.predict(X_test_scaled)
        y_score = evaluator.model.predict_proba(X_test_scaled)
        
        # Get class names
        class_names = sorted(list(set(y_test)))
        
        print("\nGenerating visualization plots...")
        # Plot visualizations
        plot_metrics(evaluator, y_test, y_pred, y_score, class_names)
        
        print("\nVisualization completed successfully!")
        
    except Exception as e:
        print(f"An error occurred in main: {str(e)}")
        print("Please ensure that:")
        print("1. The dataset path is correct")
        print("2. The dataset contains valid images")
        print("3. You have sufficient permissions to access the files")

if __name__ == "__main__":
    main() 