from flask import Flask, request, jsonify
import torch
import os
import logging
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor
import threading

# Import the model classes from the provided scripts
from speech_emotion_recognition import SpeechEmotionRecognizer
from therapeutic_response_system import TherapeuticResponseGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask apps
emotion_app = Flask("emotion_recognition")
therapy_app = Flask("therapy_response")

# Global variables for models
emotion_recognizer = None
therapy_generator = None

# Model initialization functions
def init_emotion_model():
    global emotion_recognizer
    logger.info("Initializing Speech Emotion Recognition model...")
    emotion_recognizer = SpeechEmotionRecognizer()
    emotion_recognizer.setup_model()
    # Load pretrained weights if available
    if os.path.exists("emotion_model_weights.pt"):
        emotion_recognizer.model.load_state_dict(
            torch.load("emotion_model_weights.pt")
        )
    logger.info("Speech Emotion Recognition model initialized")

def init_therapy_model():
    global therapy_generator
    logger.info("Initializing Therapeutic Response model...")
    therapy_generator = TherapeuticResponseGenerator()
    therapy_generator.initialize_model()
    # Load pretrained weights if available
    if os.path.exists("therapy_model_weights.pt"):
        therapy_generator.model.load_state_dict(
            torch.load("therapy_model_weights.pt")
        )
    logger.info("Therapeutic Response model initialized")

# Emotion Recognition endpoints
@emotion_app.route('/health', methods=['GET'])
def emotion_health():
    return jsonify({"status": "healthy", "service": "emotion_recognition"})

@emotion_app.route('/analyze', methods=['POST'])
def analyze_emotion():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    try:
        audio_file = request.files['audio']
        filename = secure_filename(audio_file.filename)
        temp_path = os.path.join("/tmp", filename)
        audio_file.save(temp_path)
        
        result = emotion_recognizer.analyze_audio(temp_path)
        os.remove(temp_path)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error analyzing audio: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Therapy Response endpoints
@therapy_app.route('/health', methods=['GET'])
def therapy_health():
    return jsonify({"status": "healthy", "service": "therapy_response"})

@therapy_app.route('/generate', methods=['POST'])
def generate_response():
    if not request.json or 'message' not in request.json:
        return jsonify({"error": "No message provided"}), 400
    
    try:
        message = request.json['message']
        response = therapy_generator.generate_response(message)
        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return jsonify({"error": str(e)}), 500

def run_emotion_service():
    init_emotion_model()
    emotion_app.run(host='0.0.0.0', port=5001)

def run_therapy_service():
    init_therapy_model()
    therapy_app.run(host='0.0.0.0', port=5002)

if __name__ == "__main__":
    # Create temporary directory for audio files if it doesn't exist
    os.makedirs("/tmp", exist_ok=True)
    
    # Start both services in separate threads
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(run_emotion_service)
        executor.submit(run_therapy_service)
