import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("Fruits Classifier with CNN")
st.write("This is a simple web app to classify images of apples, bananas, and oranges.")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model/fruits_cnn_model.h5')
    return model

model = load_model()

class_names = {0: "apple", 1: "banana", 2: "orange"}

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)
    
    def preprocess(img):
        img = img.resize((150, 150)) 
        img = np.array(img) / 255.0 
        img = np.expand_dims(img, axis=0)
        return img
    
    processed_image = preprocess(image)
    
    predictions = model.predict(processed_image)
    class_index = np.argmax(predictions[0]) 
    predicted_class = class_names[class_index]  
    
    st.write(f"Predicited class: {predicted_class}")
    st.write("Probabilities:")
    st.bar_chart(predictions[0])
