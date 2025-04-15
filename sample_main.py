from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # Import CORS
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import io
import json

# Create Flask app and set the static folder to 'public'
app = Flask(__name__, static_folder='public')

# Enable CORS for all routes
CORS(app)

# Load the model on the CPU
device = torch.device('cpu')  # Force CPU usage
model = models.resnet34(pretrained=False)
num_classes = 7  # Replace with the actual number of classes
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("skin_model2.pth", map_location=device))  # Ensure model loads on CPU
model.to(device)  # Move model to CPU
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class labels (replace with your actual class labels)
class_labels =  [
    'Acne and Rosacea Photos',
    'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
    'Eczema Photos',
    'Exanthems and Drug Eruptions',
    'Herpes HPV and other STDs Photos',
    'Melanoma Skin Cancer Nevi and Moles',
    'Nail Fungus and other Nail Disease',
    
]

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)  # Send image to CPU

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_labels[predicted.item()]
            scores = {label: score for label, score in zip(class_labels, outputs[0].tolist())}

        return jsonify({'predicted_class': predicted_class, 'scores': scores})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route to serve static files like HTML, CSS, and JS
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_static(path):
    if path != "" and path is not None:
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')  # Default to index.html if no path is specified

if __name__ == '__main__':
    app.run(debug=True)
