import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load the model
model = None

def load_model():
    global model
    model = tf.keras.models.load_model('Model2.h5')
    print("Model loaded successfully!")

# Class labels
class_names = ['Shirt', 'T-Shirt', 'Hoodies', 'Jeans', 'Shorts', 'Kurtas', 'Blazers']

@app.route('/')
def index():
    return """
    <h1>Fashion Item Classifier API</h1>
    <p>API for classifying fashion items into Hoodie, Shirt, T-shirt, or Jeans.</p>
    <h2>Endpoints:</h2>
    <ul>
        <li><code>/predict</code> - POST an image file to get classification results</li>
        <li><code>/predict/base64</code> - POST a base64 encoded image to get classification results</li>
    </ul>
    """

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read and preprocess the image
        img = Image.open(file.stream)
        img = img.resize((224, 224))  # Adjust size according to your model's input
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(class_names, predictions[0])
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/base64', methods=['POST'])
def predict_base64():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No base64 image provided'}), 400
        
        # Decode the base64 image
        image_data = data['image']
        # Remove the data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]
        
        # Decode and process the image
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize((224, 224))  # Adjust based on your model's input size
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(class_names, predictions[0])
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.before_first_request
def before_first_request():
    load_model()

if __name__ == '__main__':
    # Use PORT environment variable for compatibility with cloud platforms
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
