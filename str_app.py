import streamlit as st
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image

# Load the pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize your model
model = models.efficientnet_b0(pretrained=False)
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.classifier[1].in_features, 7)  # 7 emotion classes
)
model.load_state_dict(torch.load('best_model_1_state.pth', map_location=device))
model = model.to(device)
model.eval()

classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Streamlit UI
st.title("Live Facial Emotion Detection")
st.write("Using webcam for real-time emotion detection.")

run = st.button("Start Webcam")

if run:
    cap = cv2.VideoCapture(0)  # Start webcam capture
    if not cap.isOpened():
        st.write("Error: Unable to access the webcam.")
    
    stframe = st.empty()  # Placeholder for displaying frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_rgb_pil = Image.fromarray(face_rgb)  # Convert NumPy array to PIL image
            face_tensor = transform(face_rgb_pil).unsqueeze(0).to(device)  # Apply transformations


            with torch.no_grad():
                outputs = model(face_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)

            emotion = classes[probabilities.argmax().item()]
            confidence_percentage = probabilities.max().item() * 100

            # Draw bounding box and emotion label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion}: {confidence_percentage:.2f}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Show the frame in Streamlit
        stframe.image(frame, channels="BGR")
    
    cap.release()  # Release webcam after use
