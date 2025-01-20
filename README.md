# Ai-Dio-Help

This project integrates emotion recognition from audio input with AI-driven therapeutic responses. It leverages speech emotion recognition (SER) models and advanced AI-based chatbots to provide empathetic, therapeutic-like interactions based on the user's emotions detected from their speech. This is designed to provide users with emotional insights and supportive responses, similar to a therapist's feedback.
Project Overview

The system consists of two main components:

    Audio Emotion Recognition: A backend service using machine learning models to detect the user's emotional state from audio input (e.g., happy, sad, angry, etc.).
    AI-driven Therapeutic Response: Based on the detected emotion, the system interacts with an AI model trained to generate therapeutic, empathetic responses.

Key Features

    Emotion Recognition: Uses state-of-the-art machine learning models to analyze emotions from spoken audio.
    Therapeutic Responses: Based on the recognized emotion, an AI chatbot provides empathetic responses to help users navigate their emotions.
    Real-time Interaction: Users can upload audio files, which are processed, and they receive a therapeutic response in real time.

Technologies Used

    Backend: Node.js with Express.js
    File Upload: Multer (for handling file uploads)
    Speech Emotion Recognition Models: Hugging Face models like wav2vec2 fine-tuned on emotion datasets
    AI-based Therapy: Fine-tuned AI models for generating therapeutic responses (e.g., Llama 2/3 models)
    Audio Processing: Python with Flask and libraries like librosa for audio feature extraction
    Audio File Handling: soundfile, librosa
    Deployment: Local development or deployable to cloud services for real-time processing

Getting Started
Prerequisites

    Node.js and npm (for the server)
    Python and Flask (for the backend AI processing)
    Libraries:
        For Node.js: express, multer, node-fetch, form-data
        For Python: flask, flask_cors, librosa, soundfile, numpy

Installation
1. Clone the repository

git clone https://github.com/yourusername/audio-emotion-analyzer.git
cd audio-emotion-analyzer

2. Set up the Node.js server

    Navigate to the server directory.

cd server

    Install Node.js dependencies.

npm install

3. Set up the Python backend (for AI and emotion analysis)

    Navigate to the audio-processor directory.

cd audio-processor

    Install Python dependencies.

pip install -r requirements.txt

Running the Application
1. Start the Python server

python audio_processor.py

This will start the Flask server at http://localhost:5000, which handles the audio processing and emotion analysis.
2. Start the Node.js server

node server.js

This will start the Node.js server at http://localhost:3000, which serves the front-end and handles audio file uploads.
Uploading and Processing Audio

Once both servers are running:

    Go to http://localhost:3000 and upload an audio file (in any common format, e.g., .wav).
    The audio file will be processed by the Python backend, where the emotion will be analyzed, and a therapeutic response will be generated based on the detected emotion.

AI Model Details
1. Emotion Recognition:

    Model: Hugging Face's wav2vec2 models, fine-tuned on emotion datasets (RAVDESS, IEMOCAP).
    Emotions Detected: Happy, sad, angry, neutral, and more.
    Technology: Speech-to-Text and Audio Feature Extraction.

2. Therapeutic Responses:

    Model: Fine-tuned versions of Llama 2/3 models trained for generating empathetic responses.
    Use Case: Based on emotional input, generate calming or supportive responses, similar to what a therapist might say.

AI Models Used

    Speech Emotion Recognition (SER)
    Therapeutic Chatbot
