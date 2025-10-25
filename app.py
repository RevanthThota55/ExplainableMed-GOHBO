"""
Brain Tumor Classification Web App
Upload MRI images and get instant AI predictions
"""

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import base64
from io import BytesIO

app = Flask(__name__)

# Configuration
MODEL_PATH = 'models/brain_tumor_resnet18.pth'
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Model
print("🤖 Loading AI model...")
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()
print("✅ Model loaded successfully!")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image):
    """Make prediction on uploaded image"""
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    
    predicted_label = CLASS_NAMES[predicted_class.item()]
    confidence_score = confidence.item() * 100
    
    # Get all probabilities
    all_probs = {}
    for i, class_name in enumerate(CLASS_NAMES):
        all_probs[class_name] = round(probabilities[0][i].item() * 100, 2)
    
    return predicted_label, confidence_score, all_probs

@app.route('/')
def home():
    """Render main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Open and process image
        image = Image.open(file.stream).convert('RGB')
        
        # Make prediction
        predicted_label, confidence, all_probs = predict_image(image)
        
        # Convert image to base64 for display
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Format class name
        class_display = {
            'glioma': 'Glioma Tumor',
            'meningioma': 'Meningioma Tumor',
            'notumor': 'No Tumor (Healthy)',
            'pituitary': 'Pituitary Tumor'
        }
        
        return jsonify({
            'success': True,
            'prediction': class_display[predicted_label],
            'confidence': round(confidence, 2),
            'probabilities': all_probs,
            'image': img_str
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 BRAIN TUMOR CLASSIFICATION WEB APP")
    print("="*60)
    print("✅ Server starting...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)