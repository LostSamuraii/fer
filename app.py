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

classes_web = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
classes_up = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Streamlit UI
st.title("Live Facial Emotion Detection")

# Initialize session state for tracking options
if 'sidebar_option' not in st.session_state:
    st.session_state.sidebar_option = "Webcam"
    
if 'run_webcam' not in st.session_state:
    st.session_state.run_webcam = False

# Sidebar for navigation
sidebar_option = st.sidebar.radio("Choose an option", ["Webcam", "Upload Image"], index=["Webcam", "Upload Image"].index(st.session_state.sidebar_option))

# Update session state based on sidebar selection
st.session_state.sidebar_option = sidebar_option

# Handle the "Webcam" option
if st.session_state.sidebar_option == "Webcam":
    st.write("Using webcam for real-time emotion detection.")
    run = st.button("Start Webcam", key="webcam_button")

    if run:
        st.session_state.run_webcam = True

    if st.session_state.run_webcam:
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

                emotion = classes_web[probabilities.argmax().item()]
                confidence_percentage = probabilities.max().item() * 100

                # Draw bounding box and emotion label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{emotion}: {confidence_percentage:.2f}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Show the frame in Streamlit
            stframe.image(frame, channels="BGR")
        
        cap.release()  # Release webcam after use

# Handle the "Upload Image" option
if st.session_state.sidebar_option == "Upload Image":
    st.write("Upload an image for emotion detection.")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        # If an image is uploaded, detect faces and predict the emotion
        img = Image.open(uploaded_image)
        img = img.convert("RGB")  # Ensure the image is in RGB format
        
        # Convert the uploaded image to a NumPy array for face detection
        img_np = np.array(img)
        
        # Convert to grayscale for face detection
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Load Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # If faces are detected, crop the first detected face
        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # Get coordinates of the first face detected
            face = img_np[y:y+h, x:x+w]  # Crop the face
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            
            # Apply the transformations
            face_tensor = transform(face_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(face_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Draw bounding box on the original image to highlight the face being sent to the model
            img_cv2 = np.array(img)
            cv2.rectangle(img_cv2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            img_pil_with_box = Image.fromarray(img_cv2)

            # Display the image with bounding box
            st.image(img_pil_with_box, caption="Image with Face Detected", use_column_width=True)

            # Display emotion prediction probabilities
            st.write("Emotion Prediction Probabilities:")
            for i, emotion in enumerate(classes_up):
                confidence_percentage = probabilities[0][i].item() * 100
                st.write(f"{emotion}: {confidence_percentage:.2f}%")
        else:
            st.write("No face detected in the image.")
