import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = models.efficientnet_b0(pretrained=False)
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.classifier[1].in_features, 7)  # 7 emotion classes
)
model.load_state_dict(torch.load('best_model_1_state.pth', map_location=device))
model = model.to(device)
model.eval()

# Emotion classes
classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Preprocessing transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Convert to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Crop and preprocess the face
        face = gray_frame[y:y + h, x:x + w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
        face_tensor = transform(face_rgb).unsqueeze(0).to(device)

        # Predict emotion probabilities
        with torch.no_grad():
            outputs = model(face_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Apply softmax to get probabilities

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Set the starting point for the text (to the right of the box)
        text_x = x + w + 10  # 10 pixels to the right of the bounding box
        text_y = y  # Align text vertically with the top of the bounding box

        # Display the probability for each emotion
        for i, prob in enumerate(probabilities[0]):
            emotion = classes[i]
            confidence_percentage = prob.item() * 100  # Convert to percentage
            text = f"{emotion}: {confidence_percentage:.2f}%"

            # Display emotion text outside the box
            cv2.putText(frame, text, (text_x, text_y + (i * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Facial Emotion Recognition', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
