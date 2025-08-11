import cv2
import numpy as np
import tensorflow as tf
import os

# --- Constants ---
MODEL_PATH = os.path.join('models', 'face_model.h5')
HAARCASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# --- Load Models ---
try:
    # Load the emotion recognition model
    emotion_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"Emotion model loaded successfully from {MODEL_PATH}")

    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
    if face_cascade.empty():
        raise IOError(f"Failed to load Haar Cascade from {HAARCASCADE_PATH}")
    print(f"Haar Cascade loaded successfully from {HAARCASCADE_PATH}")

except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# --- Initialize Webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started successfully. Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Extract the face region of interest (ROI) from the grayscale frame
        roi_gray = gray_frame[y:y+h, x:x+w]

        # Preprocess the ROI for the emotion model
        # 1. Resize to 48x48
        resized_roi = cv2.resize(roi_gray, (48, 48))
        # 2. Normalize pixel values to [0, 1]
        normalized_roi = resized_roi.astype('float32') / 255.0
        # 3. Reshape for the model: (1, 48, 48, 1)
        processed_roi = np.expand_dims(np.expand_dims(normalized_roi, -1), 0)

        # Predict the emotion
        try:
            prediction = emotion_model.predict(processed_roi, verbose=0)

            # Get the emotion label and confidence
            max_index = np.argmax(prediction[0])
            predicted_emotion = EMOTION_LABELS[max_index]
            confidence = prediction[0][max_index]

            # Prepare the text to display
            text = f"{predicted_emotion} ({confidence:.2f})"

            # Put the text on the frame
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        except Exception as e:
            print(f"Error during prediction: {e}")
            cv2.putText(frame, "Error", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


    # Display the resulting frame
    cv2.imshow('Real-Time Emotion Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Release Resources ---
cap.release()
cv2.destroyAllWindows()
print("Webcam released and windows closed.")
