import os
import sys

import cv2
import dlib
import joblib
import numpy as np
import torch
import torch.nn as nn

if sys.platform == 'win32':
    pass


# Define the Neural Network
class DrowsinessNet(nn.Module):
    def __init__(self):
        super(DrowsinessNet, self).__init__()
        self.fc1 = nn.Linear(5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def get_base_path():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    else:
        return os.path.dirname(os.path.abspath(__file__))


# Load the trained model and the scaler
model = DrowsinessNet()
base_path = get_base_path()
model_path = os.path.join(base_path, 'drowsiness_model_smallestV3.pth')
scaler_path = os.path.join(base_path, 'scaler_smallestV3.pkl')
predictor_path = os.path.join(base_path, 'shape_predictor_68_face_landmarks.dat')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
scaler = joblib.load(scaler_path)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

ear_threshold = 0.21
max_drowsiness_score = 10
decay_rate = 0.1
frame_buffer = 0
drowsiness_score = 0
frame_count = 0


# Add the eye_aspect_ratio, mouth_aspect_ratio, etc.
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)


# Function to calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    horiz_dist = np.linalg.norm(mouth[0] - mouth[6])
    vert_dist_1 = np.linalg.norm(mouth[2] - mouth[10])
    vert_dist_2 = np.linalg.norm(mouth[4] - mouth[8])
    return (vert_dist_1 + vert_dist_2) / (2.0 * horiz_dist)


# Function to calculate Mouth Over Eye (MOE) ratio
def mouth_over_eye(mar, ear):
    return mar / ear if ear != 0 else float('inf')


# Function to calculate PUC
def calculate_puc(ear, ear_threshold, frame_buffer, consecutive_frames_threshold=3):
    return 1 if (frame_buffer >= consecutive_frames_threshold and ear < ear_threshold) else 0


def process_frame(frame):
    global frame_buffer, drowsiness_score, frame_count
    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        leftEye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
        rightEye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])
        mouth = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 60)])

        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
        mar = mouth_aspect_ratio(mouth)
        moe = mouth_over_eye(mar, ear)

        # Update frame buffer for PUC
        if ear < ear_threshold:
            frame_buffer += 1
        else:
            frame_buffer = 0

        puc = calculate_puc(ear, ear_threshold, frame_buffer)

        # Calculate features and use the model for prediction
        features = np.array([[ear, mar, puc, moe, frame_count]])
        scaled_features = scaler.transform(features)
        features_tensor = torch.tensor(scaled_features, dtype=torch.float32)

        with torch.no_grad():
            outputs = model(features_tensor)
            _, predicted = torch.max(outputs, 1)
            model_prediction = predicted.numpy()[0]

        # Update drowsiness score based on model prediction
        if model_prediction == 1:
            drowsiness_score -= decay_rate if drowsiness_score > 0 else 0
        else:
            drowsiness_score += 1 if drowsiness_score < max_drowsiness_score else 0

        # Display the drowsiness level on the frame
        drowsiness_level = int(drowsiness_score)
        cv2.putText(frame, f'Drowsiness Level: {drowsiness_level}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame, int(drowsiness_score)
