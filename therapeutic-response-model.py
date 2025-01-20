
"""
Datasets used:
1. Counseling and Therapy Dialogues Dataset: 
   https://github.com/behavioral-data/Empathy-Mental-Health/blob/master/dataset/sample_input_ER.csv
   - 10K+ therapy conversations
   - Multiple therapeutic approaches
   - Licensed under MIT license

2. EmpatheticDialogues:
   https://github.com/facebookresearch/EmpatheticDialogues
   - 24K conversations grounded in emotional situations
   - Download using: dataset = load_dataset("empathetic_dialogues")

3. Mental Health Conversational Data:
    https://www.kaggle.com/datasets/elvis23/mental-health-conversational-data
   - 3.4K+ therapy-like conversations
   - Multiple mental health topics
"""
# therapeutic_response_system.py

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from datetime import datetime
import json
from tqdm import tqdm
from collections import defaultdict

class TherapyDataset(Dataset):
    def __init__(self, conversations, tokenizer, max_length=128):
        self.input_ids = []
        self.attention_masks = []
        
        for conv in conversations:
            encodings = tokenizer(
                conv,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )
            self.input_ids.append(encodings['input_ids'].squeeze())
            self.attention_masks.append(encodings['attention_mask'].squeeze())
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx]
        }

class TherapeuticResponseGenerator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.max_length = 128
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.output_dir = f"therapy_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Metrics tracking
        self.training_stats = {
            'train_loss': [],
            'eval_loss': [],
            'response_lengths': [],
            'empathy_scores': [],
            'response_quality': []
        }

    def load_and_preprocess_data(self, data_path):
        """
        Load and preprocess the therapy conversation dataset
        """
        self.logger.info("Loading and preprocessing data...")
        
        # Load dataset
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Process conversations
        processed_conversations = []
        for conv in data:
            formatted = f"Client: {conv['client']}\nTherapist: {conv['therapist']}"
            processed_conversations.append(formatted)
        
        # Plot data statistics
        self.plot_conversation_statistics(processed_conversations)
        
        return train_test_split(processed_conversations, test_size=0.1, random_state=42)

    def initialize_model(self):
        """
        Initialize the GPT-2 model and tokenizer
        """
        self.logger.info("Initializing model and tokenizer...")
        
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.to(self.device)
        
        return self.model, self.tokenizer

    def train_model(self, train_data, eval_data, epochs=3, batch_size=8, learning_rate=2e-5):
        """
        Train the therapeutic response model
        """
        self.logger.info("Starting model training...")
        
        # Create datasets
        train_dataset = TherapyDataset(train_data, self.tokenizer, self.max_length)
        eval_dataset = TherapyDataset(eval_data, self.tokenizer, self.max_length)
        
        # Create dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
        
        # Initialize optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=total_steps
        )
        
        # Training loop
        for epoch in range(epochs):
            self.logger.info(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training phase
            train_loss = self._train_epoch(train_dataloader, optimizer, scheduler)
            self.training_stats['train_loss'].append(train_loss)
            
            # Evaluation phase
            eval_loss = self._evaluate_epoch(eval_dataloader)
            self.training_stats['eval_loss'].append(eval_loss)
            
            # Plot training progress
            self.plot_training_progress()
            
            self.logger.info(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")
        
        # Save model
        self.save_model()
        
        # Final visualizations
        self.plot_final_training_stats()

    def _train_epoch(self, dataloader, optimizer, scheduler):
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            # Prepare data
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        return total_loss / len(dataloader)

    def _evaluate_epoch(self, dataloader):
        """
        Evaluate for one epoch
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                total_loss += outputs.loss.item()
        
        return total_loss / len(dataloader)

    def generate_response(self, client_input, max_length=150):
        """
        Generate therapeutic response for client input
        """
        self.model.eval()
        
        # Prepare input
        prompt = f"Client: {client_input}\nTherapist:"
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Therapist:")[-1].strip()
        
        # Analyze response
        self._analyze_response_quality(response)
        
        return response

    def _analyze_response_quality(self, response):
        """
        Analyze the quality of generated response
        """
        # Calculate response length
        self.training_stats['response_lengths'].append(len(response.split()))
        
        # Calculate empathy score (basic implementation)
        empathy_phrases = ['I understand', 'I hear you', 'That sounds', 'It must be']
        empathy_score = sum(phrase.lower() in response.lower() for phrase in empathy_phrases)
        self.training_stats['empathy_scores'].append(empathy_score)
        
        # Overall quality score (basic implementation)
        quality_score = len(response.split()) / 100 + empathy_score
        self.training_stats['response_quality'].append(quality_score)

    def plot_conversation_statistics(self, conversations):
        """
        Plot statistics about the conversation dataset
        """
        plt.figure(figsize=(12, 6))
        
        # Length distribution
        lengths = [len(conv.split()) for conv in conversations]
        plt.subplot(1, 2, 1)
        sns.histplot(lengths, bins=30)
        plt.title('Conversation Length Distribution')
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        
        # Save plot
        plt.savefig(os.path.join(self.plots_dir, 'conversation_stats.png'))
        plt.close()

    def plot_training_progress(self):
        """
        Plot training and evaluation loss
        """
        plt.figure(figsize=(12, 6))
        
        plt.plot(self.training_stats['train_loss'], label='Training Loss')
        plt.plot(self.training_stats['eval_loss'], label='Evaluation Loss')
        
        plt.title('Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(self.plots_dir, 'training_progress.png'))
        plt.close()

    def plot_final_training_stats(self):
        """
        Plot final training statistics
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss curves
        axes[0, 0].plot(self.training_stats['train_loss'], label='Training Loss')
        axes[0, 0].plot(self.training_stats['eval_loss'], label='Evaluation Loss')
        axes[0, 0].set_title('Training and Evaluation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Response length distribution
        sns.histplot(self.training_stats['response_lengths'], bins=30, ax=axes[0, 1])
        axes[0, 1].set_title('Response Length Distribution')
        axes[0, 1].set_xlabel('Number of Words')
        
        # Empathy scores
        axes[1, 0].plot(self.training_stats['empathy_scores'])
        axes[1, 0].set_title('Empathy Scores Over Time')
        axes[1, 0].set_xlabel('Response Number')
        axes[1, 0].set_ylabel('Empathy Score')
        
        # Response quality
        axes[1, 1].plot(self.training_stats['response_quality'])
        axes[1, 1].set_title('Response Quality Over Time')
        axes[1, 1].set_xlabel('Response Number')
        axes[1, 1].set_ylabel('Quality Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'final_stats.png'))
        plt.close()

    def save_model(self):
        """
        Save the trained model and tokenizer
        """
        self.model.save_pretrained(os.path.join(self.output_dir, 'model'))
        self.tokenizer.save_pretrained(os.path.join(self.output_dir, 'tokenizer'))

def main():
    # Initialize the generator
    generator = TherapeuticResponseGenerator()
    
    # Load and preprocess data
    data_path = "path_to_your_therapy_dataset.json"  # Replace with your dataset path
    train_data, eval_data = generator.load_and_preprocess_data(data_path)
    
    # Initialize model
    generator.initialize_model()
    
    # Train model
    generator.train_model(train_data, eval_data)
    
    # Test generation
    test_inputs = [
        "I've been feeling really anxious lately",
        "I'm having trouble with my relationships",
        "Everything feels overwhelming"
    ]
    
    print("\nGenerating sample responses:")
    for input_text in test_inputs:
        response = generator.generate_response(input_text)
        print(f"\nClient: {input_text}")
        print(f"Therapist: {response}")

if __name__ == "__main__":
    main()