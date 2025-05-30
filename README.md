# AI-Model Project

## Overview

This repository provides a comprehensive pipeline for **student verification** and **vehicle access control**, combining **QR-code scanning** and **real-time license-plate detection**. It demonstrates an end-to-end workflow that:

1. Scans a student's QR code to verify identity.
2. Scans the student's vehicle license plate to grant access.

## Features

* **QR-code scanning** from a local image database (`TestData/qrcodes`) to look up student information in a demo database.
* **Real-time video capture** and inference via webcam.
* **License-plate detection** using a pretrained YOLOv8 model.
* **Optical Character Recognition (OCR)** integration for extracting plate numbers with PaddleOCR.
* **Interactive web front-ends** for both QR-code scanning and license-plate detection.
* **Jupyter notebook** walkthrough Yolov8 training process.

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/trungleminh2909/AI-Model---Project.git
   cd AI-Model---Project
   ```

2. **Set up a Python virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   Copy the template:

   ```bash
   cp .env.example .env
   ```

   * Set `CAMERA_INDEX` (integer, default is `0` for primary webcam).
   * Adjust any other settings as needed.

## Workflow

To run the full student verification and access control flow:

1. **QR-Code Scanning**

   * Open `qr_scan.html` in your browser (e.g., via `python -m http.server 8000`).
   * Select or upload a QR-code image from `TestData/qrcodes/`.
   * The system decodes the QR code, looks up student data in the demo database, and displays the student's name and ID.

2. **License-Plate Detection**

   * After verifying the student, open `plate_detect.html` in your browser (same server).
   * Allow webcam access to start real-time video.
   * The system detects the vehicle's license plate, performs OCR to read the plate number, and displays it on screen.

3. **Desktop Application (Optional)**

   * Run the integrated application via:

     ```bash
     python Camera.py
     ```
   * Press `q` to toggle between **QR scanning** and **license-plate detection** modes in the same window.

## Usage Examples

### QR-Code Scanning Front-End (`qr_scan.html`)

1. Launch a local HTTP server:

   ```bash
   python -m http.server 8000
   ```
2. Navigate to `http://localhost:8000/qr_scan.html`.
3. Upload a QR-code image to verify a student.

### License-Plate Detection Front-End (`plate_detect.html`)

1. With the server running, go to `http://localhost:8000/plate_detect.html`.
2. Grant webcam access.
3. Hold up the vehicle license plate to the camera to detect and read the number.

### Jupyter Notebook Walkthrough (`license-recognition.ipynb`)

Run:

```bash
jupyter notebook license-recognition.ipynb
```

Follow sections:

* **Setup**: Load models and dependencies.
* **Image Examples**: Test detection on stored images.
* **Video Demo**: Annotate video files.
* **Performance Metrics**: Evaluate speed and OCR accuracy.

## Project Structure

```
.
├── TestData/
│   └── qrcodes/                # Sample QR-code images for student lookup
├── Camera.py                   # Desktop app: toggles QR and plate modes
├── YOLOv8_plate_full.pt        # Pretrained YOLOv8 model weights
├── license-recognition.ipynb   # Notebook for detailed training pipeline walkthrough
├── plate_detect.html           # Web demo for license-plate detection
├── qr_scan.html                # Web demo for QR-code scanning & student lookup
├── requirements.txt            # Python dependencies
├── .env.example                # Template for environment variables
├── .env                        # User-specific settings (gitignored)
└── .gitignore                  # Files and directories to ignore in Git
```
