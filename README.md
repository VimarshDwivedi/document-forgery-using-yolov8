# Document Forgery Detection Using YOLOv8

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Project Overview

This repository implements a document forgery detection system using the YOLOv8 object detection model. The system analyzes document images to detect signs of forgery, such as tampering, alterations, or fake stamps. The model is trained on a custom dataset of authentic and forged documents and can be used to automate forgery detection for official documents, certificates, and IDs.

---

## Features

- Detect forged regions in document images with bounding boxes.
- Use the latest YOLOv8 architecture for high accuracy and speed.
- Easy-to-use interface for training, inference, and evaluation.
- Streamlit-based web app interface for real-time document forgery detection.
- Supports uploading images/documents for quick forgery checks.

---

## Dataset

The model is trained on a dataset consisting of labeled document images. The dataset contains:

- **Authentic documents** - scanned or photographed genuine documents.
- **Forged documents** - images containing tampered text, forged signatures, fake stamps, or alterations.

Each image has annotated bounding boxes around forged regions to guide the model during training.

---

## Model Architecture

YOLOv8 (You Only Look Once, version 8) is a state-of-the-art, single-stage object detection model that balances detection accuracy and speed. It detects objects by predicting bounding boxes and class probabilities directly from the image in a single forward pass.

Key benefits:

- Real-time inference speed.
- Robust detection with fewer false positives.
- Supports transfer learning with custom datasets.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/VimarshDwivedi/document-forgery-using-yolov8.git
cd document-forgery-using-yolov8
pip install -r requirements.txt



Requirements include:

Python 3.8+

ultralytics (YOLOv8 implementation)

OpenCV

Streamlit (for web app)

PyTorch (for model training/inference)

Usage
1. Training the model
Make sure your dataset is prepared and organized according to the YOLO format with images and corresponding .txt annotation files.

Run the training script:

bash
Copy
Edit
python train.py --data data.yaml --epochs 50 --batch 16 --img 640
data.yaml contains dataset paths and class names.

Adjust epochs, batch, and img size as per your environment.

2. Running inference on images
Use the inference script to detect forgery in a new image:

bash
Copy
Edit
python detect.py --weights runs/train/weights/best.pt --source path_to_image.jpg --conf 0.25
--weights points to your trained model weights.

--source can be a single image or a directory of images.

--conf sets the confidence threshold for detection.

3. Using the Streamlit web app
Launch the Streamlit interface:

bash
Copy
Edit
streamlit run app.py
Upload document images via the UI and get forgery detection results in real-time.

How It Works
The YOLOv8 model takes input images and processes them through its convolutional backbone to extract features.

It predicts bounding boxes and class probabilities for possible forged regions.

The predictions are filtered by confidence threshold to keep only high-confidence forgery detections.

Detected forged areas are visualized with bounding boxes and labels on the original document image.

The Streamlit app provides an interactive interface to upload and test documents easily.

Evaluation Metrics
The model is evaluated using standard object detection metrics:

Precision — Accuracy of forgery predictions.

Recall — Ability to find all forged regions.

mAP (mean Average Precision) — Overall performance combining precision and recall.

Inference time — Speed of detection per image.

Results
The trained YOLOv8 model achieves promising results in detecting forged document regions with high precision and recall. The bounding boxes correctly localize alterations, tampering, and fake stamps on document images.

Example result:


Contributing
Contributions are welcome! Feel free to:

Report issues or bugs.

Suggest improvements or new features.

Submit pull requests with bug fixes or enhancements.

Please make sure to follow the repository's coding style and include tests where applicable.


