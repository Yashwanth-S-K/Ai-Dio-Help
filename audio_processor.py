# audio_processor.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import librosa
import numpy as np
from datetime import datetime
import soundfile as sf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Configure upload folder
UPLOAD_FOLDER = 'processed_audio'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def process_audio(file_path):
    """
    Process the audio file and extract features
    """
    try:
        # Load the audio file
        y, sr = librosa.load(file_path)
        
        # Extract features
        # 1. Duration
        duration = librosa.get_duration(y=y, sr=sr)
        
        # 2. Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # 3. Average volume (RMS Energy)
        rms = librosa.feature.rms(y=y)
        average_volume = float(np.mean(rms))
        
        # 4. Pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = float(np.mean(pitches[magnitudes > np.max(magnitudes)/2]))
        
        # 5. Spectral Features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_mean = float(np.mean(spectral_centroids))
        
        return {
            'duration': round(duration, 2),
            'tempo': round(float(tempo), 2),
            'average_volume': round(average_volume, 4),
            'average_pitch': round(pitch_mean, 2),
            'spectral_centroid': round(spectral_mean, 2),
            'sample_rate': sr,
            'processed_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise

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
        
        logger.info(f"Processing file: {filename}")
        
        # Process the audio file
        results = process_audio(file_path)
        
        # Add file information to results
        results['filename'] = filename
        results['file_path'] = file_path
        
        # Clean up the file after processing
        os.remove(file_path)
        
        return jsonify({
            'success': True,
            'message': 'Audio processed successfully',
            'analysis': results
        })
    
    except Exception as e:
        logger.error(f"Error in process_audio_file: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)