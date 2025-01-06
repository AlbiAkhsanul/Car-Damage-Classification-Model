import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageDraw
import os
import numpy as np
import shutil

def load_and_preprocess_image(img):
    try:
        img = img.convert('RGB')  # Ensure image is in RGB format
        img = img.resize((224, 224))  # Resize to the input shape expected by the model
        img_array = img_to_array(img)  # Convert image to array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = preprocess_input(img_array)  # Preprocess the image
        return img_array
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def predict_image(model, img_array):
    if img_array is not None:
        predictions = model.predict(img_array)
        return predictions
    else:
        return None

# Load YOLOv8 model
model_yolo = YOLO('runs/detect/train26(best2)/weights/best.pt')
model_classification = tf.keras.models.load_model('D:/Datasets/riset_infor_dataset/model_dataset/damage_detection_mobilenet.keras')

# Streamlit app
st.title("Car Damage Detection and Classification")
st.write("Upload an image, and the model will predict and display the results!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.title("Detecting damages...")

    # Save uploaded image temporarily
    temp_image_path = "temp_image.jpg"
    image.save(temp_image_path)
    
    # Define a fixed folder for temporary files
    TEMP_FOLDER = "temp_results"

    # Ensure the folder exists and is clean
    if os.path.exists(TEMP_FOLDER):
        shutil.rmtree(TEMP_FOLDER)  # Delete existing folder and its contents
    os.makedirs(TEMP_FOLDER, exist_ok=True)  # Recreate the folder

    # Perform prediction with YOLO
    results = model_yolo(temp_image_path)
    results_for_display = model_yolo.predict(
        source=temp_image_path,
        save=True,
        project=TEMP_FOLDER,  # Fixed project name
        name="single_run",  # Fixed name to prevent multiple folders
    )
    
    # Get the result image path
    result_display_folder = results_for_display[0].save_dir
    result_for_display_image_path = os.path.join(result_display_folder, os.listdir(result_display_folder)[0])

    # Display the result image
    with Image.open(result_for_display_image_path) as result_image:
        st.image(result_image, caption="Predicted Image", use_container_width=True)
        
    # Clean up temporary files
    shutil.rmtree(TEMP_FOLDER)
    os.remove(temp_image_path)
    
    detections = results[0].boxes  # YOLO bounding boxes
    damage_summary = []  # List to store summary of damage classifications

    if len(detections) > 0:
        st.title(f"Detected {len(detections)} damages.")
        st.divider()
        for i, box in enumerate(detections):
            st.title(f"Damage Region {i+1}")
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Perbesar kotak deteksi sebesar 10%
            width = x2 - x1
            height = y2 - y1
            margin_w = int(width * 0.2)  # Tambahkan 20% dari lebar
            margin_h = int(height * 0.2)  # Tambahkan 20% dari tinggi

            # Koordinat baru dengan margin
            x1 = max(0, x1 - margin_w)
            y1 = max(0, y1 - margin_h)
            x2 = min(image.width, x2 + margin_w)
            y2 = min(image.height, y2 + margin_h)

            # Crop the region of interest
            cropped_image = image.crop((x1, y1, x2, y2))  # Crop the enlarged region

            # Display the cropped region
            st.image(cropped_image, caption=f"Damage Region {i+1}", use_container_width=True)

            # Classify the cropped region
            st.write(f"### Classifying damage in region {i+1}...")
            img_array = load_and_preprocess_image(cropped_image)
            predictions = predict_image(model_classification, img_array)

            if predictions is not None:
                # Daftar label
                labels = ["Minor", "Moderate", "Severe"]

                # Temukan indeks dari nilai terbesar
                max_index = np.argmax(predictions[0])

                # Tampilkan label dan probabilitas dengan nilai terbesar
                damage_type = labels[max_index]
                confidence = predictions[0][max_index]
                st.write(f"### Prediction: {damage_type} ({confidence:.2f})")
                
                # Tambahkan ke summary
                damage_summary.append(f"### Region {i+1}: {damage_type} ({confidence:.2f})")
            else:
                st.write("### Classification failed for this region.")
                damage_summary.append(f"### Region {i+1}: Classification failed.")

            # Add a divider for clarity
            st.divider()
        
        # Display damage summary
        st.write("### Damage Summary")
        for damage in damage_summary:
            st.write(damage)
    else:
        st.title("No damages detected. Please try uploading a different image.")
