"""
Bi-LSTM based Language Model for Burmese Text
Written by Ye Kyaw Thu, LU Lab., Myanmar.
Last updated: 15 Jan 2025.

This script implements a Bi-LSTM based language model with options for training, testing, and text generation. 
It is designed for educational purposes.

How to run:

Training:
python laphet.py --model_type bilstm --train --data ./data/train/train.txt \
--model ./model/bilstm.model --seq_len 50 --epochs 50 --batch_size 64 --lr 5e-5 \
--embed_dim 512 --hidden_dim 1024 \
--num_layers 3 --dropout 0.3 --embedding_method nn.Embedding

Text Generation:
python laphet.py --model_type bilstm --generate --model ./model/bilstm.model --seq_l  10 \
--prompt "ရဲ" --no_of_generation 10 --temperature 12 --embedding_method nn.Embedding

Batch Text Generation from File:
python laphet.py --model_type bilstm --generate --model ./model/bilstm.model --seq_len 3 \
--input start_words.txt --no_of_generation 10 --output generated_texts.txt --temperature 10 \
--embedding_method nn.Embedding


Testing:
python laphet.py --model_type bilstm --test --model ./model/bilstm.model --test_file ./data/train/test.txt \
--seq_len 50 --batch_size 64 --embedding_method nn.Embedding

"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import random
from lm_lib.util import BurmeseDataset, calculate_perplexity, calculate_cross_entropy

# Bi-LSTM Model
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, embedding_method="nn.Embedding", fasttext_model_path=None, tokenizer=None):
        super(BiLSTM, self).__init__()
        
        if embedding_method == "nn.Embedding":
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        elif embedding_method in ["fasttext_freeze", "fasttext_no_freeze"]:
            if not fasttext_model_path:
                raise ValueError("FastText model path must be provided for FastText embeddings.")
            
            from gensim.models.fasttext import load_facebook_model
            fasttext_model = load_facebook_model(fasttext_model_path)
            fasttext_vectors = fasttext_model.wv
            
            # Create a matrix of embeddings for the vocabulary
            fasttext_embeddings = torch.zeros((vocab_size, embed_dim))
            for token, idx in tokenizer.token_to_id.items():
                if token in fasttext_vectors:
                    fasttext_embeddings[idx] = torch.tensor(fasttext_vectors[token])
                else:
                    fasttext_embeddings[idx] = torch.randn(embed_dim)  # Random initialization for unknown tokens
            
            self.embedding = nn.Embedding.from_pretrained(fasttext_embeddings, freeze=(embedding_method == "fasttext_freeze"))
        else:
            raise ValueError(f"Unsupported embedding method: {embedding_method}")

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

        # Initialize weights
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)  # Xavier initialization for weights
            elif "bias" in name:
                nn.init.constant_(param, 0)  # Zero initialization for biases

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        logits = self.fc(lstm_out)
        return logits

    def generate(self, input_ids, max_length, temperature=1.0, top_k=10, top_p=0.9):
        self.eval()
        generated = input_ids.clone()
        device = next(self.parameters()).device

        # Initialize hidden states
        batch_size = input_ids.size(0)
        hidden_state = (
            torch.zeros(self.lstm.num_layers * 2, batch_size, self.lstm.hidden_size, device=device),
            torch.zeros(self.lstm.num_layers * 2, batch_size, self.lstm.hidden_size, device=device)
        )

        for _ in range(max_length):
            # Pass through the model
            embedded = self.embedding(generated[:, -1:])
            lstm_out, hidden_state = self.lstm(embedded, hidden_state)
            logits = self.fc(lstm_out[:, -1, :]) / max(temperature, 1e-6)

            # Top-k sampling
            if top_k:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, -1, None]] = -float("Inf")

            # Top-p sampling
            if top_p:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_logits[sorted_indices_to_remove] = -float("Inf")
                logits = torch.gather(sorted_logits, -1, sorted_indices)

            # Fallback if all logits are -inf
            if torch.all(logits == -float("Inf")):
                print("All logits are -Inf after filtering! Fallback to raw logits.")
                logits = self.fc(lstm_out[:, -1, :]) / max(temperature, 1e-6)

            # Convert logits to probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

            # Debugging: Validate probabilities
            if torch.any(torch.isnan(probabilities)) or torch.any(probabilities < 0):
                print(f"Invalid probabilities detected! Logits: {logits}")
                raise ValueError("Probabilities contain invalid values.")

            # Sample the next token
            next_token = torch.multinomial(probabilities, num_samples=1)

            # Append next token
            generated = torch.cat((generated, next_token), dim=1)

        return generated

# Training Function
def train_bilstm(model, train_dataset, dev_dataset, epochs, batch_size, lr, device, model_save_path, embedding_method, fasttext_model_path):
    """
    Train a Bidirectional LSTM (BiLSTM) model on the given dataset with validation monitoring.

    :param model: BiLSTM model to be trained
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
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    best_val_loss = float('inf')  # Track the best validation loss

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Training loop
        for input_ids, attention_mask in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} (Training)"):
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            outputs = model(input_ids)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = input_ids.view(-1)
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
            for input_ids, attention_mask in tqdm(dev_dataloader, desc=f"Epoch {epoch+1}/{epochs} (Validation)"):
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                outputs = model(input_ids)
                outputs = outputs.view(-1, outputs.size(-1))
                targets = input_ids.view(-1)
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
                "hidden_dim": model.lstm.hidden_size,  # Save model hyperparameters
                "num_layers": model.lstm.num_layers,  # Save model hyperparameters
                "dropout": model.lstm.dropout,  # Save model hyperparameters
                "seq_len": train_dataset.seq_len,  # Save sequence length
                "embedding_method": embedding_method,  # Save embedding method
                "fasttext_model_path": fasttext_model_path # Save FastText model path
            }
            torch.save(checkpoint, model_save_path)
            print(f"Best model saved at {model_save_path} with validation loss: {best_val_loss:.4f}")

# Test Function
def test_bilstm(model, tokenizer, test_file, seq_len, batch_size, device):
    """
    Evaluate the model on a test set and calculate perplexity and cross-entropy.
    :param model: Trained BiLSTM model
    :param tokenizer: Tokenizer object
    :param test_file: Path to the test dataset
    :param seq_len: Sequence length
    :param batch_size: Batch size
    :param device: Device (cpu or cuda)
    """
    # Load the test dataset
    with open(test_file, "r", encoding="utf-8") as f:
        texts = f.read().splitlines()
    
    dataset = BurmeseDataset(texts, tokenizer, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    model.to(device)

    total_perplexity = 0.0
    total_cross_entropy = 0.0
    count = 0

    with torch.no_grad():
        for input_ids, attention_mask in tqdm(dataloader, desc="Testing"):
            input_ids = input_ids.to(device)
            logits = model(input_ids)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # Calculate metrics for the batch
            perplexity = calculate_perplexity(log_probs, input_ids)
            cross_entropy = calculate_cross_entropy(log_probs, input_ids)

            total_perplexity += perplexity
            total_cross_entropy += cross_entropy
            count += 1

    avg_perplexity = total_perplexity / count
    avg_cross_entropy = total_cross_entropy / count

    print(f"Average Perplexity on Test Data: {avg_perplexity:.4f}")
    print(f"Average Cross-Entropy on Test Data: {avg_cross_entropy:.4f}")
