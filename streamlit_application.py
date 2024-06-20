import streamlit as st
from PIL import Image
import torch

# Load yolov5 pre-trained model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


# Build streamlit interface
st.title("Image Component Analysis")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)
    st.write("")

    if st.button('Analyse Image'):


        # Provide image to the model
        results = model(image)

        # Process predictions
        detected_objects = results.pandas().xyxy[0]['name'].tolist()

        # Print image components
        st.write("Identified Components:")
        for component in detected_objects:
            st.write(component)
