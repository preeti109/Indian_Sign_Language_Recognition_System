from keras.models import load_model
import cv2
import numpy as np

# Load the trained model
try:
    model = load_model("sign_language_model.h5")  # Replace with your model's file path
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Define class labels (adjust as per your model's classes)
label = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
                'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 
                'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def extract_features(image):
    """
    Preprocess the image: resize to the correct input size, ensure 3 channels, normalize, and reshape for the model.
    """
    # Check the model's expected input shape
    print(f"Expected model input shape: {model.input_shape}")  # Debugging the expected input size
    
    # Resize the image to match the expected input size of the model
    expected_input_size = model.input_shape[1:3]  # Get the height and width from the input shape
    image = cv2.resize(image, (expected_input_size[1], expected_input_size[0]))  # Resize to model input size
    
    # Convert to RGB if the image has only one channel (grayscale)
    if len(image.shape) == 2:  # If grayscale (1 channel), convert to RGB (3 channels)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    feature = np.array(image)
    feature = feature.reshape(1, expected_input_size[0], expected_input_size[1], 3)  # Resize for model's input
    return feature / 255.0  # Normalize pixel values to range [0, 1]

# Start the webcam feed
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()  # Capture frame from webcam
    if not _:
        print("Failed to capture frame.")
        break
    
    # Define region of interest (ROI) for hand gesture
    cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)  # Draw rectangle on screen for ROI
    cropframe = frame[40:300, 0:300]  # Crop the hand gesture region from the frame
    
    # Preprocess the cropped frame to match the model's input size
    cropframe = extract_features(cropframe)
    
    # Predict the gesture from the model
    pred = model.predict(cropframe)  # Predict the class
    
    # Print raw predictions and max prediction for debugging
    print(f"Raw Prediction: {pred}")  # Debugging the raw prediction
    print(f"Max Prediction: {np.max(pred)}")  # Debugging the confidence
    
    prediction_label = label[pred.argmax()]  # Get the class with the highest prediction
    accuracy = "{:.2f}".format(np.max(pred) * 100)  # Calculate accuracy percentage
    
    # Display the prediction on the frame
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)  # Draw top rectangle for displaying text
    if prediction_label == 'blank':
        cv2.putText(frame, " ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, f'{prediction_label}  {accuracy}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Show the output frame with prediction
    cv2.imshow("output", frame)
    
    # Exit the loop when 'ESC' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for the ESC key
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
