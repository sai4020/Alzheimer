import os
import numpy as np
from flask import Flask, request, render_template
from keras.preprocessing import image
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

# Initialize Flask App
app = Flask(__name__)

# Load the model
model = load_model('adp.h5')

@app.route('/', methods=['GET'])
def index():
    return render_template('home.html')

@app.route('/predict1', methods=['GET'])
def predict1():
    return render_template('innerpage.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file uploaded", 400

    f = request.files['image']
    basepath = os.path.dirname(__file__)
    uploads_path = os.path.join(basepath, 'uploads')
    os.makedirs(uploads_path, exist_ok=True)  # Ensure uploads folder exists
    file_path = os.path.join(uploads_path, secure_filename(f.filename))
    
    try:
        f.save(file_path)
    except Exception as e:
        return f"Error saving file: {str(e)}", 500

    # Preprocess the image
    try:
        img = image.load_img(file_path, target_size=(180, 180))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0  # Normalize

        # Predict
        prediction = model.predict(x)
        label = np.argmax(prediction, axis=1)[0]  # Get class index

        # Map predictions to labels
        labels = {
            0: "Mild Demented",
            1: "Moderate Demented",
            2: "Non Demented",
            3: "Very Mild Demented"
        }
        result = labels.get(label, "Unknown Prediction")
    except Exception as e:
        return f"Error processing the image: {str(e)}", 500

    return render_template('innerpage.html', result=result)

if __name__ == "__main__":
    app.run(port=4000, debug=False)
