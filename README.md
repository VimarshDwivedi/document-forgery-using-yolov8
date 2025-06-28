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
