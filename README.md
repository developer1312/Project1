# Real-Time Facial Emotion Recognition

This project is a Python application that performs real-time facial emotion recognition using a webcam.

## Overview

The application captures your webcam feed, uses an OpenCV Haar Cascade classifier to detect faces, and then predicts the emotion for each face using a pre-trained TensorFlow model. The predicted emotion is then displayed on the video feed.

## Features

- Real-time video processing.
- Face detection.
- 7-class emotion classification (angry, disgust, fear, happy, sad, surprise, neutral).
- Visual feedback with bounding boxes and labels.

## Setup

First, clone the repository. Then, install the necessary dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Execute the following command from the project's root directory:

```bash
python emotion_recognition_webcam.py
```

A window will appear showing the webcam feed. Press 'q' to quit the application.

## Dependencies

- tensorflow==2.13.0
- opencv-python==4.8.1.78
- numpy==1.24.3
