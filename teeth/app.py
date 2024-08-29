import streamlit as st
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image
import numpy as np

model = load_model('my_model.h5')

class_names = {
    0: 'CaS',
    1: 'CoS',
    2: 'Gum',
    3: 'MC',
    4: 'OC',
    5: 'OLP',
    6: 'OT'
}
def preprocess_image(image):
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

st.title("Image Classification with a PreTrained Model")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)

    st.image(image, caption='Uploaded Image.', use_column_width=True)

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction)

    predicted_class_name = class_names.get(predicted_class_index, 'Unknown Class')

    st.write(f"Prediction: {predicted_class_name}")
    st.write(f"Confidence: {prediction[0][predicted_class_index]:.4f}")

    st.write("Class Probabilities:")
    for i, class_name in class_names.items():
        st.write(f"{class_name}: {prediction[0][i]:.4f}")
