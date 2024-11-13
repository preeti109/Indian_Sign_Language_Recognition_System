from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model = load_model("sign_language_model.h5")
print("Model loaded successfully!")

# Define class labels
class_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    img_file = request.files['image']
    img = Image.open(io.BytesIO(img_file.read()))
    
    # Preprocess the image
    img = img.convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    predicted_label = class_labels[predicted_class]
    
    # Return the prediction as JSON
    return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
