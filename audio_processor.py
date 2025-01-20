from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Configure upload folder
UPLOAD_FOLDER = 'processed_audio'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# URLs of the models running in the background
EMOTION_RECOGNIZER_URL = "http://localhost:5001/analyze"
THERAPEUTIC_RESPONSE_URL = "http://localhost:5002/generate"


@app.route('/process-audio', methods=['POST'])
def process_audio_file():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        file = request.files['audio']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the uploaded file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"audio_{timestamp}.wav"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        logger.info(f"Uploaded file saved: {filename}")

        # Send the file to the emotion recognition model
        with open(file_path, 'rb') as audio_file:
            emotion_response = requests.post(
                EMOTION_RECOGNIZER_URL, files={'audio': audio_file}
            )
            if emotion_response.status_code != 200:
                logger.error(f"Emotion recognition failed: {emotion_response.json()}")
                return jsonify({'error': 'Emotion recognition failed'}), 500

            emotion_data = emotion_response.json()

        # Extract the detected emotion
        emotion = emotion_data.get('emotion')
        if not emotion:
            logger.error("No emotion detected in the response")
            return jsonify({'error': 'No emotion detected'}), 500

        logger.info(f"Emotion detected: {emotion}")

        # Send the detected emotion to the therapeutic response model
        therapeutic_response = requests.post(
            THERAPEUTIC_RESPONSE_URL, json={'message': f"I've been feeling {emotion} lately"}
        )
        if therapeutic_response.status_code != 200:
            logger.error(f"Therapeutic response generation failed: {therapeutic_response.json()}")
            return jsonify({'error': 'Therapeutic response generation failed'}), 500

        therapeutic_data = therapeutic_response.json()

        # Clean up the file after processing
        os.remove(file_path)

        # Return the final result
        return jsonify({
            'success': True,
            'emotion': emotion_data,
            'therapeutic_response': therapeutic_data
        })

    except Exception as e:
        logger.error(f"Error in process_audio_file: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
