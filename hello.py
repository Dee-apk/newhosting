from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import gdown
# https://drive.google.com/file/d/1THtiQb4-AX0TTDvdrq_IVUnXFsPsZazA/view?usp=drive_link
# Load the trained model

MODEL_URL = 'https://drive.google.com/uc?export=download&id=1THtiQb4-AX0TTDvdrq_IVUnXFsPsZazA'

# Path to save the downloaded model
MODEL_PATH = 'cat_dog_classifier.h5'

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = tf.keras.models.load_model(MODEL_PATH)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images/'

# Route for homepage (UI)
@app.route('/')
def index():
    image_path = 'kity.png'
    return render_template('index.html',image_path=image_path)
 
# Route for uploading and classifying image
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Load image and preprocess it
        img = Image.open(file.stream)
        img = img.resize((150, 150))  # Resize image to 150x150
        img = np.array(img) / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Predict the class of the image
        prediction = model.predict(img)
        result = 'Dog' if prediction[0] > 0.5 else 'Cat'

        return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)

