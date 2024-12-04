from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# Initialize Flask app and allow CORS for frontend interaction
app = Flask(__name__)
CORS(app)

# Load pre-trained EfficientNet model for emotion detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0(pretrained=False)
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.classifier[1].in_features, 7)  # 7 emotion classes
)
model.load_state_dict(torch.load("best_model_1_state.pth", map_location=device))
model.to(device)
model.eval()

# Define emotion classes
classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Define image preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# API endpoint for image uploads
@app.route('/predict', methods=['POST'])
def predict_emotion():
    if 'image' not in request.files:
        return jsonify({"error": "No image file found"}), 400

    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert('RGB')  # Read and convert to RGB

    # Preprocess image
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    # Prepare response
    predictions = {classes[i]: float(probabilities[0][i]) * 100 for i in range(len(classes))}
    top_emotion = classes[probabilities.argmax().item()]
    response = {"top_emotion": top_emotion, "predictions": predictions}

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
