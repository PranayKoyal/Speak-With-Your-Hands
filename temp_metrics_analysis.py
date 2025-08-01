import sys
import os
from pathlib import Path

# Get the absolute path to the evaluation script
current_dir = Path(__file__).parent
eval_script_path = current_dir / "sign_language_website" / "backend" / "evaluation" / "sign_language_ml_evaluation.py"

# Add the directory containing the script to Python path
sys.path.append(str(current_dir))

# Import the SignLanguageEvaluator class from the evaluation script
from sign_language_ml_evaluation import SignLanguageEvaluator
import matplotlib.pyplot as plt

def display_metrics():
    # Initialize evaluator
    evaluator = SignLanguageEvaluator()
    
    # Download and prepare dataset
    if evaluator.download_dataset():
        # Load and preprocess data
        if evaluator.load_and_preprocess_data():
            # Train and evaluate model
            accuracy, conf_matrix, class_report = evaluator.train_and_evaluate()
            
            # The metrics will be automatically displayed by the evaluator
            # including:
            # - Accuracy score
            # - Classification report (precision, recall, F1-score)
            # - Confusion matrix visualization
            # - Feature importance plot

if __name__ == "__main__":
    display_metrics() 