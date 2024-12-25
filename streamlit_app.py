import streamlit as st
import tensorflow as tf
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('VGG16_model.h5')
IMG_SIZE = (224, 224)

# Title of the application
st.title('Disease Predictor')

with st.expander("Help Section", expanded=False):
        st.write("### Instructions:")
        st.write("1. Click 'Browse files' to upload an image.")
        st.write("2. Ensure the image is in JPG, JPEG, or PNG format.")
        st.write("3. Wait for the prediction to be displayed.")
        st.write("4. The predicted class and confidence will appear below the uploaded image.")

        st.write("### Sample Images and Descriptions:")
        col1, col2, col3 = st.columns(3)
        Width=150
        with col1:
            st.image("/workspaces/Disease-Detection/S1.jpg", caption="Blood Cancer", width=Width)
        with col2:
            st.image("/workspaces/Disease-Detection/S2.jpg", caption="Skin Cancer", width=Width)
        with col3:
            st.image("/workspaces/Disease-Detection/S3.jpg", caption="Kidney Stone", width=Width)
        col4, col5, col6 = st.columns(3)
        with col4:
            st.image("/workspaces/Disease-Detection/S4.jpg", caption="Pneumonia", width=Width)
        with col5:
            st.image("/workspaces/Disease-Detection/S5.jpg", caption="Brain Tumor", width=Width)
        with col6:
            st.image("/workspaces/Disease-Detection/S6.jpg", caption="Diabetic Retinopathy", width=Width)

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Process the uploaded image
    img = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=IMG_SIZE)
    st.image(img, caption='Uploaded Image', width=300)

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100

    # Class labels
    class_labels = [
        'No Blood Cancer', 'No Skin Cancer', 'Diabetic Retinopathy', 'Kidney Stone', 'Skin Cancer',
        'No Diabetic Retinopathy', 'No Kidney Stone',
        'Pre-B Blood Cancer', 'Pro-B Blood Cancer', 'Early Pre-B Blood Cancer', 'Glioma Brain Tumor',
        'Meningioma Brain Tumor', 'No Pneumonia',
        'No Brain Tumor', 'Pituitary Brain Tumor', 'Pneumonia'
    ]

    # Display predictions
    st.write(f"Predicted Class: {class_labels[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}%")
