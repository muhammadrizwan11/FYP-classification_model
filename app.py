import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

# Load the trained model
model = tf.keras.models.load_model('FYP-classification_model.h5')

st.title('MRI Image Classification')
st.write('Upload an MRI image and the model will predict whether it is significant or notsignificant.')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    img_array = np.array(image)
    
    # Ensure the image has 3 channels (RGB)
    if len(img_array.shape) == 2:  # grayscale image
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA image
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    img_resized = cv2.resize(img_array, (256, 256))
    img_normalized = img_resized / 255.0
    img_reshaped = img_normalized.reshape((1, 256, 256, 3))

    # Ensure the reshaped array has the correct shape
    st.write(f"Image shape: {img_reshaped.shape}")

    # Make prediction
    prediction = model.predict(img_reshaped)
    prediction_class = 'significant' if prediction[0][0] > 0.5 else 'notsignificant'

    st.write(f'The model predicts that the image is: **{prediction_class}**')
