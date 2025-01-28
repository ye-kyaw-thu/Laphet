"""
BERT Language Model for Burmese Text
Written by Ye Kyaw Thu, LU Lab., Myanmar.
Last updated: 17 Jan 2025.

This script implements a BERT-based language model with options for training, testing, and text generation. 
It is designed for educational purposes.

How to run:

for training:
time python laphet.py --model_type bert --train --data ./data/train.txt --model ./model/bert.model \
--seq_len 50 --epochs 10 --batch_size 32 --lr 0.0001 --embedding_method nn.Embedding

for text generation:
time python laphet.py --model_type bert --generate \
--model ./model/bert.model --seq_len 50 --prompt "ရဲ" \
--no_of_generation 10 --embedding_method nn.Embedding

for text generation from file:
time python laphet.py --model_type bert --generate --model ./model/bert.model \
--seq_len 2 --input start_words.txt \
--no_of_generation 5 --output generated_texts.txt --embedding_method nn.Embedding

for testing:
time python laphet.py --model_type bert --test --model ./model/bert.model \
--test_file ./data/train/test.txt --seq_len 50 --batch_size 64 --embedding_method nn.Embedding

"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import random
from lm_lib.util import BurmeseDataset, calculate_perplexity, calculate_cross_entropy
from gensim.models.fasttext import load_facebook_model

# BERT Model
class BERT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, seq_len, embedding_method="nn.Embedding", fasttext_model_path=None, tokenizer=None):
        super(BERT, self).__init__()
        
        if embedding_method == "nn.Embedding":
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        elif embedding_method in ["fasttext_freeze", "fasttext_no_freeze"]:
            if not fasttext_model_path:
                raise ValueError("FastText model path must be provided for FastText embeddings.")
            
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

        self.position_embedding = nn.Embedding(seq_len, embed_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, attention_mask):
        positions = torch.arange(0, x.size(1)).unsqueeze(0).to(x.device)
        x = self.embedding(x) + self.position_embedding(positions)
    
        # Transpose attention_mask to match the expected shape
        attention_mask = attention_mask.transpose(0, 1)
    
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=(attention_mask == 0))
        return self.fc(x)

    def generate(self, input_ids, max_length, temperature=1.0, top_k=None, top_p=None):
        self.eval()
        generated = input_ids.clone()
        for _ in range(max_length):
            outputs = self.forward(generated, attention_mask=(generated != 0).long())
            logits = outputs[:, -1, :]  # Get logits of the last token
        
            # Scale logits by temperature
            logits = logits / max(temperature, 1e-6)  # Avoid division by zero
        
            # Top-k sampling
            if top_k:
                values, _ = torch.topk(logits, top_k)
                min_values = values[:, -1].unsqueeze(-1)
                logits[logits < min_values] = -float('Inf')
        
            # Top-p sampling
            if top_p:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_logits[sorted_indices_to_remove] = -float('Inf')
                logits = torch.gather(sorted_logits, -1, sorted_indices)
        
            # Clamp logits to avoid invalid values
            logits = torch.clamp(logits, min=-1e9, max=1e9)
        
            # Convert logits to probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
            # Validate probabilities
            if torch.any(torch.isnan(probabilities)) or torch.any(probabilities < 0):
                raise ValueError("Probabilities contain invalid values.")
        
            # Sample the next token
            next_token = torch.multinomial(probabilities, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
    
        return generated

# Training Function
def train_bert(model, train_dataset, dev_dataset, epochs, batch_size, lr, device, model_save_path, embedding_method, fasttext_model_path):
    """
    Train the BERT model on the given dataset with validation monitoring.

    :param model: BERT model to be trained
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
            outputs = model(input_ids, attention_mask)
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
                outputs = model(input_ids, attention_mask)
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
                "num_heads": model.layers[0].self_attn.num_heads,  # Access num_heads from self_attn
                "num_layers": len(model.layers),  # Save model hyperparameters
                "ff_dim": model.layers[0].linear1.out_features,  # Save feedforward dimension
                "seq_len": train_dataset.seq_len,  # Save sequence length
                "embedding_method": embedding_method,  # Save embedding method
                "fasttext_model_path": fasttext_model_path # Save FastText model path
            }
            torch.save(checkpoint, model_save_path)
            print(f"Best model saved at {model_save_path} with validation loss: {best_val_loss:.4f}")

# Testing Function
def test_bert(model, tokenizer, test_file, seq_len, batch_size, device):
    """
    Evaluate the BERT model on test data and calculate perplexity and cross-entropy.
    :param model: Trained BERT model
    :param tokenizer: Tokenizer for input text
    :param test_file: Path to the test dataset
    :param seq_len: Sequence length for evaluation
    :param batch_size: Batch size for evaluation
    :param device: Device to run the evaluation
    """
    # Load test data
    with open(test_file, "r", encoding="utf-8") as f:
        texts = f.read().splitlines()

    # Create dataset and dataloader
    dataset = BurmeseDataset(texts, tokenizer, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    model.to(device)

    total_perplexity = 0.0
    total_cross_entropy = 0.0
    total_batches = 0

    with torch.no_grad():
        for input_ids, attention_mask in tqdm(dataloader, desc="Testing"):
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            
            # Forward pass to compute logits
            logits = model(input_ids, attention_mask)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # Calculate metrics for the batch
            perplexity = calculate_perplexity(log_probs, input_ids)
            cross_entropy = calculate_cross_entropy(log_probs, input_ids)

            total_perplexity += perplexity
            total_cross_entropy += cross_entropy
            total_batches += 1

    avg_perplexity = total_perplexity / total_batches
    avg_cross_entropy = total_cross_entropy / total_batches

    print(f"Average Perplexity on Test Data: {avg_perplexity:.4f}")
    print(f"Average Cross-Entropy on Test Data: {avg_cross_entropy:.4f}")
