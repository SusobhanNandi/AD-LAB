from flask import Flask, request, render_template, send_from_directory
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = load_model('C:/Users/susob/OneDrive/Desktop/model/cat_dog_model .h5')

# Directory to save uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# CSV file to save predictions
CSV_FILE = 'predictions.csv'
if not os.path.exists(CSV_FILE):
    # Create the CSV file with headers if it doesn't exist
    pd.DataFrame(columns=["Filename", "Prediction"]).to_csv(CSV_FILE, index=False)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    uploaded_file_path = None

    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files or request.files['file'].filename == '':
            return render_template('index.html', prediction="No file uploaded!")

        file = request.files['file']

        # Save the file temporarily
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Preprocess the image
        img = cv2.imread(file_path)
        img = cv2.resize(img, (256, 256))  # Match model input shape
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Make prediction
        prediction_prob = model.predict(img)
        prediction = 'Dog' if prediction_prob[0][0] > 0.5 else 'Cat'

        # Save prediction to CSV
        df = pd.read_csv(CSV_FILE)
        new_entry = pd.DataFrame([[file.filename, prediction]], columns=["Filename", "Prediction"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)

        uploaded_file_path = file_path

    return render_template('index.html', prediction=prediction, image_path=uploaded_file_path)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
