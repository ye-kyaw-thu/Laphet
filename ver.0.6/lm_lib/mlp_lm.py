"""
MLP Language Model for Burmese Text
Written by Ye Kyaw Thu, LU Lab., Myanmar.
Last updated: 20 Jan 2025.

This script implements a MLP-based language model with options for training, testing, and text generation. 
Designed for educational applications with the Burmese language.

How to run:

Training:
time python laphet.py --model_type mlp --train --data ./data/train.txt --model ./model/mlp.model \
--seq_len 50 --epochs 10 --batch_size 32 --lr 0.0001

Text Generation:
time python laphet.py --model_type --generate --model ./model/mlp.model --seq_len 50 --prompt "ရဲ" \
--no_of_generation 10

Batch Text Generation from File:
time python laphet.py --model_type --generate --model ./model/mlp.model --seq_len 2 \
--input start_words.txt --no_of_generation 5 --output generated_texts.txt

Testing:
time python laphet.py --model_type --test --model ./model/mlp.model \
--test_file ./data/test.txt --seq_len 50 --batch_size 64

"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn.functional as F
import random
from lm_lib.util import BurmeseDataset, calculate_perplexity, calculate_cross_entropy

# MLP-based Language Model
class MLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, seq_len):
        super(MLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )

    def forward(self, x):
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        outputs = []
        for i in range(x.size(1)):  # Iterate over the sequence length
            token_embedding = embedded[:, i, :]  # Extract embeddings for token i
            logits = self.mlp(token_embedding)  # Pass through MLP
            outputs.append(logits)  # Collect outputs
        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, vocab_size)
        return outputs

    def generate(self, input_ids, max_length, temperature=1.0, top_k=None, top_p=None):
        self.eval()
        generated = input_ids.clone()
        device = input_ids.device

        for _ in range(max_length):
            outputs = self.forward(generated[:, -self.mlp[0].in_features:])  # Use the last `seq_len` tokens
            logits = outputs[:, -1, :]  # Logits for the next token
            logits = logits / temperature  # Apply temperature scaling

            # Check for NaN or Inf in logits
            if torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits)):
                logits = torch.zeros_like(logits)  # Replace invalid logits with zeros

            if top_k:
                # Apply top-k filtering
                values, _ = torch.topk(logits, top_k)
                min_values = values[:, -1].unsqueeze(-1)
                logits[logits < min_values] = -float('Inf')

            if top_p:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_logits[sorted_indices_to_remove] = -float('Inf')
                logits = torch.scatter(logits, 1, sorted_indices, sorted_logits)

            # Ensure logits validity after filtering
            if torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits)):
                logits = torch.zeros_like(logits)

            probabilities = F.softmax(logits, dim=-1)

            # Ensure probabilities validity
            if torch.any(torch.isnan(probabilities)) or torch.any(torch.isinf(probabilities)):
                raise ValueError("Probabilities contain NaN or Inf values.")

            next_token = torch.multinomial(probabilities, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)

        return generated


# Training function
def train_mlp(model, train_dataset, dev_dataset, epochs, batch_size, lr, device, model_save_path):
    """
    Train a Multi-Layer Perceptron (MLP) model on the given dataset with validation monitoring.

    :param model: MLP model to be trained
    :param train_dataset: Training dataset containing tokenized input sequences and attention masks
    :param dev_dataset: Validation dataset for monitoring overfitting
    :param epochs: Number of training epochs
    :param batch_size: Batch size for training
    :param lr: Learning rate for the optimizer
    :param device: Device to run the training (e.g., 'cuda' or 'cpu')
    :param model_save_path: Path to save the best model based on validation loss
    """

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    model.to(device)

    best_val_loss = float('inf')  # Track the best validation loss

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Training loop
        for inputs, targets in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} (Training)"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in tqdm(dev_dataloader, desc=f"Epoch {epoch+1}/{epochs} (Validation)"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(dev_dataloader)
        print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}")

        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save the model and tokenizer information
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "token_to_id": train_dataset.tokenizer.token_to_id,  # Save tokenizer vocabulary
                "id_to_token": train_dataset.tokenizer.id_to_token,  # Save tokenizer vocabulary
                "embed_dim": model.embedding.embedding_dim,  # Save model hyperparameters
                "hidden_dim": model.mlp[0].out_features,  # Save model hyperparameters
                "seq_len": train_dataset.seq_len  # Save sequence length
            }
            torch.save(checkpoint, model_save_path)
            print(f"Best model saved at {model_save_path} with validation loss: {best_val_loss:.4f}")

# Test function
def test_mlp(model, tokenizer, test_file, seq_len, batch_size, device):
    """
    Evaluate the model on a test dataset and calculate perplexity and cross-entropy.
    :param model: Trained MLP model
    :param tokenizer: Tokenizer instance
    :param test_file: Path to the test dataset
    :param seq_len: Sequence length
    :param batch_size: Batch size for testing
    :param device: Device to run the evaluation on
    """
    # Load test dataset
    with open(test_file, "r", encoding="utf-8") as f:
        texts = f.read().splitlines()
    dataset = BurmeseDataset(texts, tokenizer, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model.eval()
    total_perplexity = 0.0
    total_cross_entropy = 0.0
    num_batches = 0

    with torch.no_grad():
        for input_ids, target_ids in tqdm(dataloader, desc="Testing"):
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)

            # Forward pass
            logits = model(input_ids)  # [batch_size, seq_len, vocab_size]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # Convert to log probabilities

            # Calculate metrics for the batch
            perplexity = calculate_perplexity(log_probs, target_ids)
            cross_entropy = calculate_cross_entropy(log_probs, target_ids)

            total_perplexity += perplexity
            total_cross_entropy += cross_entropy
            num_batches += 1

    avg_perplexity = total_perplexity / num_batches
    avg_cross_entropy = total_cross_entropy / num_batches

    print(f"Average Perplexity on Test Data: {avg_perplexity:.4f}")
    print(f"Average Cross-Entropy on Test Data: {avg_cross_entropy:.4f}")

