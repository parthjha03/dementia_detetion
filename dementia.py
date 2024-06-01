import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model_path = "D:\segmentation\hackathon website\dementia detection\dementia.h5" #your model path
model = load_model(model_path)

# Function to preprocess image for dementia detection
def preprocess_image(image_path):
    input_image = cv.imread(image_path)
    input_image_gs = cv.cvtColor(input_image, cv.COLOR_RGB2GRAY)
    input_image_gs = cv.resize(input_image_gs, (100, 100))
    input_image_gs = input_image_gs / 255
    input_image_gs = input_image_gs.reshape(1, 100, 100, 1)
    return input_image_gs

# Define class names
class_names = ['Non-Detected', 'Mild-Demented', 'Moderate-Demented']

# Preprocess the image and make prediction
def predict_dementia(image_path):
    input_image = preprocess_image(image_path)
    prediction = model.predict(input_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name

# Call the predict function with the image path provided as command-line argument
if __name__ == "__main__":
    import sys

    # image_path = "your image path"
    predicted_class = predict_dementia(image_path)
    print(predicted_class)
