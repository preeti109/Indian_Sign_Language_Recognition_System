from keras.models import load_model
import cv2
import numpy as np

# Load the trained model
try:
    model = load_model("sign_language_model.h5")  # Replace with your model path
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Define class labels
class_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
                'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 
                'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def predict_image(image_path):
    """
    Preprocess the input image and predict its class using the trained model.
    """
    try:
        # Load and preprocess the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB if necessary
        resized_image = cv2.resize(image, (128, 128))  # Resize to match model input size
        normalized_image = resized_image / 255.0       # Normalize pixel values to [0, 1]
        input_image = np.expand_dims(normalized_image, axis=0)  # Add batch dimension

        print(f"Processed image shape: {input_image.shape}")  # Debug input shape

        # Predict
        predictions = model.predict(input_image)
        print(f"Predictions: {predictions}")  # Debug raw prediction values

        predicted_class = np.argmax(predictions)
        confidence = predictions[0][predicted_class]
        print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")

        return class_labels[predicted_class] if confidence > 0.5 else "Uncertain"
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Example usage
if __name__ == "__main__":
    image_path = "23.jpg"  # Replace with your test image path
    prediction = predict_image(image_path)
    if prediction:
        print(f"Predicted class: {prediction}")
    else:
        print("Prediction failed.")
