import gradio as gr
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the model and labels
model = load_model("Model.h5", compile=False)

with open("labels.txt", "r") as file:
    class_names = file.readlines()

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Function to process the image and make a prediction
def classify_image(image):
    # Resize and normalize the image
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Make the prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    return class_name, confidence_score

# Define the Gradio interface
iface = gr.Interface(
    fn=classify_image,  # The function to run when the API is called
    inputs=gr.Image(type="pil"),  # Input type: Image
    outputs=["text", "number"],  # Output: class and confidence score
    live=True  # Enable live updates while selecting an image
)

# Launch the Gradio app
iface.launch()
