# Driver Drowsiness Detection Web App

A full-stack web application for detecting driver drowsiness from uploaded driving videos. The system uses computer vision and deep learning techniques to analyze facial landmarks and predict drowsiness states, then generates an output video with alerts for risky moments.

## Overview

This project combines:
- a FastAPI backend for video upload, processing, and status tracking
- a React + Vite frontend for user interaction and result visualization
- OpenCV, MediaPipe, and TensorFlow/Keras-based inference
- frame-by-frame video analysis with drowsiness detection output

## Features

- Upload driving videos through a modern web interface
- Process videos for drowsiness detection in the background
- Track job progress and processing status
- View proof frames and download the processed output video
- Support optional custom model and beep files
- Handle large video uploads with background processing

## Tech Stack

### Backend
- Python
- FastAPI
- Uvicorn
- OpenCV
- MediaPipe
- TensorFlow / Keras
- NumPy
- MoviePy

### Frontend
- React
- Vite
- JavaScript / JSX
- CSS

## Project Structure

- backend/main.py - FastAPI routes, file upload handling, and startup checks
- backend/processor.py - Video processing and drowsiness detection pipeline
- backend/utils.py - Helper functions for preprocessing and facial metrics
- frontend/src/ - React application UI and components
- backend/assets/ - Required model and resource files

## Prerequisites

Make sure you have the following installed:

- Python 3.10 or later
- Node.js 18 or later
- A trained .h5 drowsiness detection model
- A dlib landmark predictor file: shape_predictor_68_face_landmarks.dat

You can place the required files in:
- backend/assets/
- or the project root

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd share
```

### 2. Backend setup

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Frontend setup

```bash
cd ../frontend
npm install
```

## Running the Application

### Start the backend

```bash
cd backend
python main.py
```

The backend will run at:
- http://localhost:8000

### Start the frontend

```bash
cd frontend
npm run dev
```

The frontend will run at:
- http://localhost:5173

## API Endpoints

- POST /upload - Upload a video file and optional model/beep files
- GET /status/{job_id} - Check processing status
- GET /video/{job_id} - Download the processed output video
- GET /frame/{job_id}/{filename} - Retrieve proof frames

## Usage

1. Open the frontend in your browser.
2. Upload a video file.
3. Optionally upload your own model (.h5) and beep file (.mp3).
4. Start the processing job.
5. Monitor the progress and view the final output video.

## Notes

- The backend supports large video uploads up to 500MB.
- Processing runs in the background to keep the interface responsive.
- The app is intended for research, academic, and demonstration purposes.

## Demo / Project Goal

The goal of this project is to build a practical driver monitoring system that can help detect fatigue and raise awareness of dangerous driving behavior in real time or through recorded video analysis.
