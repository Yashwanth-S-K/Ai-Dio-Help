
"""
Datasets used:
1. RAVDESS Dataset: https://zenodo.org/record/1188976
   - 24 professional actors (12 female, 12 male)
   - 7 emotional expressions: calm, happy, sad, angry, fearful, surprise, and disgust
   - Download using: dataset = load_dataset("DataJedi/ravdess_emotional_speech")

2. IEMOCAP Dataset: https://sail.usc.edu/iemocap/
   - 10 actors (5 female, 5 male)
   - Emotional expressions: angry, happy, sad, neutral
   - Note: Requires application for access
"""
# speech_emotion_recognition.py

import torch
import torchaudio
import numpy as np
import pandas as pd
from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Audio as IPythonAudio
import os
from datetime import datetime
import logging

class SpeechEmotionRecognizer:
    def __init__(self, model_name="facebook/wav2vec2-base", num_labels=4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.num_labels = num_labels
        self.feature_extractor = None
        self.model = None
        self.trainer = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.output_dir = f"emotion_recognition_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'plots'), exist_ok=True)
        
        self.emotion_mapping = {
            "angry": 0,
            "happy": 1,
            "neutral": 2,
            "sad": 3
        }

    def load_data(self):
        """Load and prepare the RAVDESS dataset"""
        self.logger.info("Loading RAVDESS dataset...")
        dataset = load_dataset("DataJedi/ravdess_emotional_speech")
        
        # Filter for specific emotions
        dataset = dataset.filter(lambda x: x["emotion"] in self.emotion_mapping.keys())
        
        # Convert emotions to labels
        dataset = dataset.map(lambda x: {"label": self.emotion_mapping[x["emotion"]]})
        
        # Split dataset
        train_test = dataset["train"].train_test_split(test_size=0.2)
        
        self.visualize_data_distribution(dataset["train"])
        return train_test["train"], train_test["test"]

    def setup_model(self):
        """Initialize the model and feature extractor"""
        self.logger.info("Setting up model and feature extractor...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_name,
            return_attention_mask=True
        )
        
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            classifier_dropout=0.1
        ).to(self.device)

    def preprocess_data(self, examples):
        """Preprocess audio data"""
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = self.feature_extractor(
            audio_arrays,
            sampling_rate=16000,
            padding=True,
            max_length=16000,
            truncation=True,
            return_tensors="pt"
        )
        return inputs

    def train(self, train_dataset, eval_dataset, num_epochs=5):
        """Train the model"""
        self.logger.info("Starting model training...")
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=3e-5,
            weight_decay=0.01,
            push_to_hub=False,
            logging_dir=os.path.join(self.output_dir, 'logs')
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics
        )
        
        # Train the model
        train_result = self.trainer.train()
        
        # Plot training metrics
        self.plot_training_history()
        
        return train_result

    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return {
            "accuracy": accuracy_score(eval_pred.label_ids, predictions),
            "classification_report": classification_report(eval_pred.label_ids, predictions)
        }

    def analyze_audio(self, audio_path):
        """Analyze a single audio file"""
        self.logger.info(f"Analyzing audio file: {audio_path}")
        
        # Load and preprocess audio
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Plot waveform
        self.plot_waveform(waveform[0].numpy(), 16000, audio_path)
        
        # Get prediction
        inputs = self.feature_extractor(waveform[0].numpy(), sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(predictions, dim=1).item()
        
        # Get results
        reverse_mapping = {v: k for k, v in self.emotion_mapping.items()}
        result = {
            "predicted_emotion": reverse_mapping[predicted_class],
            "confidence_scores": {
                emotion: predictions[0][idx].item() * 100
                for emotion, idx in self.emotion_mapping.items()
            }
        }
        
        # Plot confidence scores
        self.plot_confidence_scores(result["confidence_scores"])
        
        return result

    # Visualization Methods
    def visualize_data_distribution(self, dataset):
        """Plot distribution of emotions in dataset"""
        emotions = [example['emotion'] for example in dataset]
        plt.figure(figsize=(10, 6))
        sns.countplot(x=emotions)
        plt.title('Distribution of Emotions in Dataset')
        plt.xlabel('Emotion')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'emotion_distribution.png'))
        plt.close()

    def plot_waveform(self, waveform, sample_rate, title):
        """Plot audio waveform"""
        plt.figure(figsize=(12, 4))
        time_axis = np.arange(0, len(waveform)) / sample_rate
        plt.plot(time_axis, waveform)
        plt.title(f'Waveform: {os.path.basename(title)}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'waveform.png'))
        plt.close()

    def plot_training_history(self):
        """Plot training metrics"""
        history = self.trainer.state.log_history
        train_loss = [x['loss'] for x in history if 'loss' in x]
        eval_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label='Training Loss')
        plt.plot(range(0, len(train_loss), len(train_loss)//len(eval_loss)),
                eval_loss, label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'training_history.png'))
        plt.close()

    def plot_confidence_scores(self, scores):
        """Plot emotion confidence scores"""
        plt.figure(figsize=(10, 6))
        emotions = list(scores.keys())
        values = list(scores.values())
        plt.bar(emotions, values)
        plt.title('Emotion Recognition Confidence Scores')
        plt.xlabel('Emotion')
        plt.ylabel('Confidence (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'confidence_scores.png'))
        plt.close()

def main():
    # Initialize recognizer
    recognizer = SpeechEmotionRecognizer()
    
    # Load data
    train_dataset, eval_dataset = recognizer.load_data()
    
    # Setup model
    recognizer.setup_model()
    
    # Train model
    recognizer.train(train_dataset, eval_dataset)
    

if __name__ == "__main__":
    main()