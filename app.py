# app.py
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

app = Flask(__name__)
model = load_model("Model.h5")
classes = ['Hoodie', 'Shirt', 'T-shirt', 'Jeans']

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img = Image.open(file).resize((224, 224))  # adjust size as per your model
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    class_id = int(np.argmax(prediction))
    return jsonify({"class": classes[class_id], "confidence": float(np.max(prediction))})