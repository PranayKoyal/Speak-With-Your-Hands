# Sentence-Enhanced Sign Language Communication System

This project is a real-time sign language recognition system that uses a machine learning model to interpret hand gestures from a live webcam feed and translates them into text. The system is designed with a modern frontend and a powerful backend to provide a seamless user experience.

## Features

- **Real-Time Gesture Recognition**: Interprets sign language gestures from a live video feed in real-time.
- **Machine Learning Model**: Utilizes a Random Forest classifier to recognize a vocabulary of gestures.
- **Hand Landmark Detection**: Employs MediaPipe to detect and track hand landmarks.
- **Modern UI**: A responsive and intuitive user interface built with React.
- **Backend API**: A Flask-based backend that handles video processing and gesture recognition.

## Why Random Forest?

For the task of gesture recognition, a **Random Forest** model was chosen for several key reasons:

- **High Accuracy and Robustness**: Random Forest is an ensemble learning method that combines multiple decision trees to produce more accurate and stable predictions. This makes it less prone to overfitting compared to a single decision tree.
- **Efficiency**: It is relatively fast to train and can make predictions quickly, which is essential for a real-time application like this.
- **Handles High-Dimensional Data**: The hand landmark data from MediaPipe consists of many features (x, y, and z coordinates for 21 landmarks). Random Forest can handle this high-dimensional data effectively without requiring extensive feature selection.
- **Feature Importance**: The model can provide insights into which hand landmarks are the most important for distinguishing between different gestures, which can be valuable for model tuning and analysis.

## Project Structure

```
sign_language_website/
|-- backend/
|   |-- src/
|   |   |-- main.py           # Main backend script with Flask routes
|   |-- models/
|   |   |-- custom_gesture_model.pkl # Trained ML model
|   |   |-- scaler.pkl        # Scaler for the model features
|   |-- requirements.txt      # Python dependencies for the backend
|-- frontend/
|   |-- src/
|   |   |-- App.tsx           # Main React component
|   |   |-- ...
|   |-- package.json          # Node.js dependencies for the frontend
|-- README.md                 # This file
```

## Setup and Installation

To run this project on a new machine, you'll need to set up both the backend and the frontend.

### Prerequisites

- Python 3.9+
- Node.js v14+ and npm
- A webcam

### Backend Setup

1.  **Navigate to the backend directory**:
    ```bash
    cd 7/sign_language_website/backend
    ```

2.  **Create and activate a virtual environment**:
    - On Windows:
      ```bash
      python -m venv venv
      .\venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      python3 -m venv venv
      source venv/bin/activate
      ```

3.  **Install the required Python packages**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the backend server**:
    ```bash
    python src/main.py
    ```
    The backend should now be running at `http://localhost:5000`.

### Frontend Setup

1.  **Open a new terminal** and navigate to the frontend directory:
    ```bash
    cd 7/sign_language_website/frontend
    ```

2.  **Install the required Node.js packages**:
    ```bash
    npm install
    ```

3.  **Run the frontend development server**:
    ```bash
    npm start
    ```
    The frontend should now be running at `http://localhost:3000` and will open automatically in your browser.

## How to Use

1.  Once the frontend is open, click the "Start Video" button.
2.  Allow the browser to access your webcam.
3.  The video feed will appear, and the system will begin to recognize and display the gestures you make. 