import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


# Define class labels
class_names = [
    "Corn___Common_Rust",
    "Corn___Gray_Leaf_Spot",
    "Corn___Healthy",
    "Corn___Northern_Leaf_Blight",
    "Potato___Early_Blight",
    "Potato___Healthy",
    "Potato___Late_Blight",
    "Rice___Brown_Spot",
    "Rice___Healthy",
    "Rice___Leaf_Blast",
    "Rice___Neck_Blast",
    "Wheat___Brown_Rust",
    "Wheat___Healthy",
    "Wheat___Yellow_Rust"
]

# Define the function for image classification
def classify_image(image):
    # Preprocess the image
    image = image.resize((224, 224))  # Resize the image to match the model's input shape
    image = np.array(image)  # Convert image to numpy array
    image = image / 255.0  # Normalize the image
    
    # Reshape the image to match the model's input shape
    image = np.reshape(image, (1, 224, 224, 3))
    
    # Load the model
    model = tf.keras.models.load_model('model.h5')
    
    # Perform prediction
    predictions = model.predict(image)
    
    # Get the predicted class and confidence
    class_index = np.argmax(predictions)
    class_name = class_names[class_index]
    confidence = predictions[0][class_index] * 100
    
    return class_name, confidence

# Set page title
st.title(":red[Agricultural Crop Disease Classification]")

# Create the file uploader and submit button
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
submit_button = st.button("Predict")

if submit_button and uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image')
    
    # Classify the image
    class_name, confidence = classify_image(image)
    
    # Display the result
    st.write(f"Class: {class_name}")
    st.write(f"Confidence: {confidence:.2f}%")