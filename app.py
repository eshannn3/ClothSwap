from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
model = load_model("model.h5")
classes = ['Hoodie', 'Shirt', 'T-shirt', 'Jeans']

@app.route("/")
def index():
    return "Image classification API is running."

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img = Image.open(file).convert("RGB").resize((224, 224))  # adjust size if needed
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    class_id = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return jsonify({
        "class": classes[class_id],
        "confidence": confidence
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)