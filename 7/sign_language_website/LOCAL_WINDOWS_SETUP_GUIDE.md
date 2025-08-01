# Local Windows Setup Guide: Sign Language Recognition Website

This guide provides step-by-step instructions for setting up, training, and running the complete Sign Language Recognition website (Flask backend + React frontend) on your local Windows machine.

## Prerequisites

Before you begin, ensure you have the following installed on your Windows system:

1.  **Python:** Version 3.9 or higher (Python 3.11 recommended). You can download it from [python.org](https://www.python.org/downloads/). Make sure to check the box "Add Python to PATH" during installation.
2.  **Node.js:** Version 16 or higher. You can download it from [nodejs.org](https://nodejs.org/). This includes npm (Node Package Manager).
3.  **Git:** (Optional, but recommended for managing code). You can download it from [git-scm.com](https://git-scm.com/).
4.  **Web Browser:** Chrome, Firefox, or Edge.
5.  **Webcam:** A functional webcam connected to your computer.

## 1. Obtain Project Files

Download the `sign_language_website_local_setup.zip` file provided.
Extract the contents of the zip file to a location of your choice on your computer (e.g., `C:\Users\YourUsername\Projects\sign_language_website`).

Your project directory structure should look like this:

```
sign_language_website/
├── backend/
│   ├── src/
│   │   ├── main.py
│   │   └── train_custom_model.py  <-- Training Script
│   ├── requirements.txt
│   └── venv/  (This will be created)
├── frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── ... (React project files)
├── user_datasets/
│   ├── alphabet_test/
│   ├── alphabet_train/        <-- Example training data source
│   │   ├── A/                 <-- Class Subfolder
│   │   │   ├── img1.jpg
│   │   │   └── ...
│   │   ├── B/
│   │   │   └── ...
│   │   └── ...
│   ├── ann_sub_test/
│   ├── ann_test/
│   └── ann_test_val/
├── deployment_limitations.md
├── alternative_deployment_strategies.md
└── design_doc.md
```

## 2. Place Your Custom Datasets

Navigate to the `sign_language_website/user_datasets/` directory.
Place your custom dataset files into the corresponding subfolders.

**IMPORTANT for Training:** The training script (`train_custom_model.py`) expects the training data directory (e.g., `user_datasets/alphabet_train/`) to contain **subdirectories named after each gesture class** (e.g., `A`, `B`, `C`, `hello`, `goodbye`). Inside each class subdirectory, place the corresponding image files (.jpg, .png, .jpeg).

*   Example: Put images for the letter 'A' into `user_datasets/alphabet_train/A/`
*   Example: Put images for the letter 'B' into `user_datasets/alphabet_train/B/`

## 3. Set Up the Backend (Flask Server)

1.  **Open Command Prompt or PowerShell:** Navigate to the backend directory.
    ```bash
    cd path\to\sign_language_website\backend
    ```
    (Replace `path\to` with the actual path where you extracted the project).

2.  **Create Python Virtual Environment:**
    ```bash
    python -m venv venv
    ```
    *(If you have multiple Python versions, ensure you use the correct one, e.g., `python3.11 -m venv venv`)*

3.  **Activate Virtual Environment:**
    ```bash
    venv\Scripts\activate
    ```
    You should see `(venv)` at the beginning of your command prompt line.

4.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    This might take a few minutes as it installs libraries like Flask, OpenCV, MediaPipe, scikit-learn, etc.

## 4. Set Up the Frontend (React App)

1.  **Open another Command Prompt or PowerShell:** Navigate to the frontend directory.
    ```bash
    cd path\to\sign_language_website\frontend
    ```

2.  **Install Node.js Dependencies:**
    ```bash
    npm install
    ```
    This will install React and all other frontend libraries listed in `package.json`.

## 5. Train Your Custom Model (Optional but Recommended)

Before running the main application, you can train a gesture recognition model using your own datasets.

1.  **Ensure Backend Environment is Active:** Make sure you are in the command prompt with the `(venv)` activated (from Step 3.3).

2.  **Run the Training Script:** Execute the following command, pointing it to your training data directory.
    ```bash
    python src/train_custom_model.py --data_dir ..\user_datasets\alphabet_train
    ```
    *   Replace `..\user_datasets\alphabet_train` with the actual path to the directory containing your class subfolders (e.g., `..\user_datasets\your_combined_data`).
    *   The script will process images in the specified directory, extract hand landmarks, train a RandomForest model, and evaluate it.
    *   Training might take some time depending on the size of your dataset.

3.  **Output:** The script will save two files in the main `sign_language_website` directory (one level above `backend`):
    *   `custom_gesture_model.pkl`: The trained machine learning model.
    *   `custom_gesture_scaler.pkl`: The scaler used to preprocess the data (needed for prediction).

4.  **Command Options (Optional):**
    *   `--output_dir`: Specify a different directory to save the model/scaler (default is `..`).
    *   `--model_name`: Change the output model filename (default: `custom_gesture_model`).
    *   `--scaler_name`: Change the output scaler filename (default: `custom_gesture_scaler`).
    *   `--n_estimators`: Change the number of trees in the RandomForest (default: 100).
    *   `--test_split`: Change the proportion of data used for testing (default: 0.2 or 20%).

## 6. Run the Application

After setting up and optionally training your model, run the backend and frontend servers.

1.  **Run the Backend Server:**
    *   In the **first** command prompt (where the backend `venv` is active):
        ```bash
        python src/main.py
        ```
    *   The server will start. If `custom_gesture_model.pkl` and `custom_gesture_scaler.pkl` exist in the `sign_language_website` directory, it will load your custom model. Otherwise, it will fall back to the default model or the synthetically trained one.
    *   You should see output indicating the Flask server is running, likely on `http://127.0.0.1:5000` or `http://localhost:5000`.

2.  **Run the Frontend Server:**
    *   In the **second** command prompt (in the `frontend` directory):
        ```bash
        npm start
        ```
    *   This will start the React development server, usually on `http://localhost:3000`.
    *   It should automatically open a new tab in your default web browser.

## 7. Access the Application

*   Open your web browser and navigate to the address provided by the React development server (usually `http://localhost:3000`).
*   You should see the Sign Language Recognition website interface.
*   If you trained a custom model, the backend will now use it for gesture recognition.
*   *Note: The current frontend is a placeholder structure. Full functionality requires further development to connect components to the backend API.*

## 8. Stopping the Application

*   To stop the servers, go to each command prompt window (backend and frontend) and press `Ctrl + C`.
*   Confirm if prompted.
*   To deactivate the backend virtual environment, type `deactivate` in its command prompt.

## Troubleshooting

*   **Training Errors:**
    *   `ModuleNotFoundError`: Ensure the backend virtual environment (`venv`) is activated before running `python src/train_custom_model.py`.
    *   `No subdirectories found`: Make sure your `--data_dir` points to a folder containing subfolders named after your gesture classes (e.g., `A`, `B`, `hello`).
    *   `No images found`: Check that the class subfolders contain valid image files (.png, .jpg, .jpeg).
    *   `Insufficient data`: You need enough images across at least two different classes to train a model.
    *   Memory Errors: Large datasets might require significant RAM.
*   **`ModuleNotFoundError` (Python Runtime):** Ensure the backend virtual environment (`venv`) is activated before running `pip install` or `python src/main.py`.
*   **Camera Not Working:**
    *   Make sure your webcam is connected and not used by another application.
    *   Check browser permissions to allow camera access for `localhost`.
    *   The backend code uses `cv2.CAP_DSHOW` for better Windows compatibility, but you might need to try `cv2.VideoCapture(0)` if issues persist.
*   **Port Conflicts:** If port 5000 (backend) or 3000 (frontend) is already in use, the servers might fail to start. You can modify the port numbers in `backend/src/main.py` (e.g., `app.run(port=5001)`) and potentially configure the frontend proxy if needed (though usually not required for local development).
*   **`npm` errors:** Ensure Node.js and npm are installed correctly and added to your PATH. Try deleting the `node_modules` folder and `package-lock.json` in the `frontend` directory and running `npm install` again.
*   **Firewall Issues:** Your Windows Firewall might block the application. Ensure Python and Node.js are allowed to communicate through the firewall.

Enjoy using and customizing the Sign Language Recognition System locally!
