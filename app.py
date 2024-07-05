import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

# Load the trained model
model = load_model('FYP-classification_model.h5')

st.title('MRI Image Classification')
st.write('Upload an MRI image and the model will predict whether it is significant or notsignificant.')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (256, 256))
    img_normalized = img_resized / 255.0
    img_reshaped = img_normalized.reshape((1, 256, 256, 3))

    # Make prediction
    prediction = model.predict(img_reshaped)
    prediction_class = 'significant' if prediction[0][0] > 0.5 else 'notsignificant'

    st.write(f'The model predicts that the image is: **{prediction_class}**')
