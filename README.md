# Face Recognition Attendance System
![Face Recognition Attendance System](./Face-recognition.png)

## Overview
This repository features a Face Recognition Attendance System, a Python-based application designed for automated attendance tracking. The system identifies individuals by their facial features and records their presence, providing an efficient and modern alternative to traditional attendance methods.
The core of the system relies on pre-trained Haar Cascade classifiers for face detection and a custom-trained model for face recognition, built using a dataset of training images and corresponding labels.

## Features
* **Automated Face Detection**: Utilizes Haar Cascade classifiers to accurately detect faces within video streams or images.

* **Face Recognition**: Identifies registered individuals by comparing detected faces against a trained dataset of known faces.

* **Attendance Logging**: Records the attendance of recognized individuals, enabling automated tracking.

* **Training Module**: Includes functionality to train the face recognition model using a set of labeled images.

* **Real-time Processing**: Designed to process video feeds from a webcam for live attendance marking.

## Technologies Used
* **Python**: The primary programming language.

* **OpenCV(cv2)**: For real-time video capture, image processing, face detection (Haar Cascade), and basic image manipulation.

* **NumPy**: For numerical operations, especially array manipulation in image processing.

* **Pillow(PIL)**: For image handling, if needed for specific image operations.

## Model used:
The face recognition system employs a pipeline involving:

* **Face Detection**: Haar Cascade Classifiers are used as the initial stage to rapidly detect frontal faces within an image or video frame. This is a machine learning-based approach where a cascade function is trained from a large set of positive (faces) and negative (non-faces) images.

* **Feature Extraction & Training**: Once faces are detected, the system extracts features from these facial regions. The extracted features and corresponding labels (person IDs) are then fed into a chosen face recognition algorithm. Common algorithms used in OpenCV for this include:

* **Local Binary Patterns Histograms (LBPH) Face Recognizer**: This algorithm describes the local texture and shape of facial images. It's known for its robustness to illumination changes and its ability to handle small variations in face pose.

The trained model learns to map specific facial patterns to unique individual identities.

## Data Collection & Preparation
* **Image Acquisition**: A dataset of images is gathered for each individual intended to be recognized by the system. It's crucial to collect multiple images per person, ideally under varying conditions (different lighting, slight pose variations, expressions) to make the model more robust.

* **Labeling**: Each image is accurately labeled with a unique identifier corresponding to the individual in the image. These labels are critical for the supervised learning process.

* **Data Organization**: Images are typically organized into folders, with each folder representing a unique individual (e.g., dataset/0/, dataset/1/ for individuals with IDs 0, 1, etc.).

## Data Analysis
While not always a standalone "analysis" phase in simple systems, implicit analysis happens during:
* **Quality Check**: Reviewing collected images for clarity, proper lighting, and consistent framing.

* **Quantity Check**: Ensuring a sufficient number of images per person for effective training.

* **Haar Cascade Tuning**: Initial experimentation with Haar Cascade parameters (scale factor, min neighbors) to optimize face detection performance on the specific image set.

## Data Preprocessing
Before training, raw images undergo preprocessing:
* **Grayscale Conversion**: Color images are converted to grayscale to reduce dimensionality and focus on luminance features.

* **Face Detection & Cropping**: Using Haar Cascade classifiers, faces are detected in each training image. The detected facial regions are then cropped to ensure only the face is fed to the recognition algorithm.

* **Resizing & Normalization**: Cropped face images are resized to a uniform dimension and pixel values might be normalized to a standard range to ensure consistency for the recognition algorithm.

## Training the Model
* **Loading Data**: The preprocessed facial data (images/features) and their corresponding numerical labels are loaded.

* **Model Initialization**: An instance of the chosen face recognition model (e.g., cv2.face.LBPHFaceRecognizer_create()) is initialized.

* **Training**: The model.train() method is invoked, passing the preprocessed face data and their labels. During this phase, the model learns the unique patterns of each registered face.

* **Model Saving**: After successful training, the trained model is saved to a file (e.g., trainer/trainer.yml). This file encapsulates the learned patterns and weights, allowing the system to load it directly for recognition without needing to retrain every time.

## How to Run the Project
To view this project locally on your machine:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/sjain2580/Face-recognition-attendance-system.git
    ```

2.  **Navigate to the Project Directory:**
    ```bash
    cd Face-recognition-attendance-system
    ```

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    ```
   * **Activate the virtual environment:**
        * **Windows:** `.\venv\Scripts\activate`
        * **macOS/Linux:** `source venv/bin/activate`

4.  **Install Dependencies:**
    Make sure you have all the necessary libraries installed.
    ```bash
    pip install -r requirements.txt
    ```

5. **Prepare Training Data:**
   Create a folder (e.g., dataset/) for your training images.
   Inside dataset/, create subfolders for each individual, named with their ID (e.g., 0, 1, 2, or user_1, user_2).
   ```bash
    mkdir dataset
    ```

6. **Train the Face Recognition Model:**
   Run the training script training.py:
   ```bash
    python training.py
   ```

7. **Run the Attendance System:**
    Run the main attendance script testing.py:
   ```bash
    python testing.py
   ```

## Live Deployment
Check the live app here - https://25866n7q-5000.inc1.devtunnels.ms/

## Contributors
**https://github.com/sjain2580**

## Connect with Me
Feel free to reach out if you have any questions or just want to connect!
**[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sjain04/)**
**[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/sjain2580)**
**[![Email](https://img.shields.io/badge/-Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:sjain040395@gmail.com)**

---
