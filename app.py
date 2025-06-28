import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import numpy as np
from PIL import Image
import os
import PyPDF2
from pdf2image import convert_from_path
import io
from pathlib import Path

try:
    st.set_page_config(page_title="Document Forgery Detection", layout="centered")
    st.title("📄 Document Forgery Detection using YOLOv8")

    # Load the model
    @st.cache_resource
    def load_model():
        try:
            return YOLO("best.pt")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None

    model = load_model()
    
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload a document (PDF, JPG, PNG, JPEG)", 
        type=["pdf", "jpg", "jpeg", "png"],
        key="document_uploader"
    )
    
    # Confidence threshold slider
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.01,
        help="Adjust this value to control how confident the model needs to be to show a detection. Lower values will show more detections but may include false positives."
    )

    if uploaded_file is not None:
        try:
            # Create temporary directory for PDF processing
            temp_dir = tempfile.mkdtemp()
            temp_path = None

            # Handle different file types
            if uploaded_file.type == "application/pdf":
                st.info("Converting PDF pages to images...")
                try:
                    # Save PDF to temporary file
                    pdf_path = os.path.join(temp_dir, "temp.pdf")
                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    # Convert PDF to images
                    images = convert_from_path(pdf_path, output_folder=temp_dir)
                    
                    if len(images) > 0:
                        # Process each page
                        for i, image in enumerate(images):
                            st.subheader(f"Page {i+1} of {len(images)}")
                            
                            # Save image temporarily
                            img_path = os.path.join(temp_dir, f"page_{i+1}.jpg")
                            image.save(img_path, "JPEG")
                            
                            # Process the image with confidence threshold
                            results = model(img_path, conf=confidence_threshold)
                            
                            # Display original image
                            st.subheader("Original Page")
                            st.image(image, use_container_width=True)

                            # Display results
                            st.subheader("Detection Results")
                            for result in results:
                                boxes = result.boxes
                                if len(boxes) > 0:
                                    annotated_frame = result.plot()
                                    st.image(annotated_frame, caption='Detected Objects', use_container_width=True)
                                    st.write(f"Number of detections: {len(boxes)}")
                                    st.write("Detection Details:")
                                    for box in boxes:
                                        st.write(f"Class: {box.cls.item()}, Confidence: {box.conf.item():.2f}")
                                else:
                                    st.info("No objects detected in this page.")
                                    st.write("Try:")
                                    st.write("1. Using a clearer image")
                                    st.write("2. Making sure the image contains objects the model was trained to detect")
                                    st.write("3. Checking if the image is properly aligned and not too small")
                                    st.write("4. Adjusting the confidence threshold")
                    else:
                        st.error("No pages found in the PDF")
                except Exception as e:
                    st.error(f"Error converting PDF: {str(e)}")
                finally:
                    # Clean up temporary PDF file
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
            else:
                # For image files
                # Save uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                if model is not None:
                    # Run prediction with confidence threshold
                    results = model(temp_path, conf=confidence_threshold)
                    
                    # Display original image
                    st.subheader("Original Image")
                    image = Image.open(uploaded_file)
                    st.image(image, use_container_width=True)

                    # Display results
                    st.subheader("Detection Results")
                    for result in results:
                        boxes = result.boxes
                        if len(boxes) > 0:
                            annotated_frame = result.plot()
                            st.image(annotated_frame, caption='Detected Objects', use_container_width=True)
                            st.write(f"Number of detections: {len(boxes)}")
                            st.write("Detection Details:")
                            for box in boxes:
                                st.write(f"Class: {box.cls.item()}, Confidence: {box.conf.item():.2f}")
                        else:
                            st.info("No objects detected in the image.")
                            st.write("Try:")
                            st.write("1. Using a clearer image")
                            st.write("2. Making sure the image contains objects the model was trained to detect")
                            st.write("3. Checking if the image is properly aligned and not too small")
                            st.write("4. Adjusting the confidence threshold")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
        finally:
            # Clean up temporary files
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                os.rmdir(temp_dir)

    if uploaded_file is not None:
        try:
            # Create temporary directory for PDF processing
            temp_dir = tempfile.mkdtemp()
            temp_path = None

            # Handle different file types
            if uploaded_file.type == "application/pdf":
                st.info("Converting PDF pages to images...")
                try:
                    # Save PDF to temporary file
                    pdf_path = os.path.join(temp_dir, "temp.pdf")
                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    # Convert PDF to images
                    images = convert_from_path(pdf_path, output_folder=temp_dir)
                    
                    if len(images) > 0:
                        # Process each page
                        for i, image in enumerate(images):
                            st.subheader(f"Page {i+1} of {len(images)}")
                            
                            # Process the image
                            if model is not None:
                                results = model(img_path, conf=confidence_threshold)
                                
                                # Display original image
                                st.subheader("Original Page")
                                st.image(image, use_container_width=True)

                                # Display results
                                st.subheader("Detection Results")
                                for result in results:
                                    boxes = result.boxes
                                    if len(boxes) > 0:
                                        annotated_frame = result.plot()
                                        st.image(annotated_frame, caption='Detected Objects', use_container_width=True)
                                        st.write(f"Number of detections: {len(boxes)}")
                                        for box in boxes:
                                            st.write(f"Class: {box.cls.item()}, Confidence: {box.conf.item():.2f}")
                                    else:
                                        st.info("No objects detected in this page.")
    
                    else:
                        st.error("No pages found in the PDF")
                except Exception as e:
                    st.error(f"Error converting PDF: {str(e)}")
                finally:
                    # Clean up temporary PDF file
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
            else:
                # For image files
                # Save uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                if model is not None:
                    # Run prediction
                    results = model(temp_path, conf=confidence_threshold)
                    
                    # Display original image
                    st.subheader("Original Image")
                    image = Image.open(uploaded_file)
                    st.image(image, use_container_width=True)

                    # Display results
                    st.subheader("Detection Results")
                    for result in results:
                        boxes = result.boxes
                        if len(boxes) > 0:
                            annotated_frame = result.plot()
                            st.image(annotated_frame, caption='Detected Objects', use_container_width=True)
                            st.write(f"Number of detections: {len(boxes)}")
                            st.write("Detection Details:")
                            for box in boxes:
                                st.write(f"Class: {box.cls.item()}, Confidence: {box.conf.item():.2f}")
                        else:
                            st.info("No objects detected in the image.")
                            st.write("Try:")
                            st.write("1. Using a clearer image")
                            st.write("2. Making sure the image contains objects the model was trained to detect")
                            st.write("3. Checking if the image is properly aligned and not too small")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
        finally:
            # Clean up temporary files
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                os.rmdir(temp_dir)

except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")
