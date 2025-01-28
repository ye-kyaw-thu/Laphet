import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Tokenizer
class Tokenizer:
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}

    def fit(self, texts):
        tokens = set()
        for text in texts:
            tokens.update(text.split())
        tokens.add("[UNK]")  # Add unknown token
        tokens.add("[PAD]")  # Add padding token
        self.token_to_id = {token: idx for idx, token in enumerate(tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def encode(self, text, return_tensors=None):
        ids = [self.token_to_id.get(token, self.token_to_id["[UNK]"]) for token in text.split()]
        if return_tensors == "pt":
            return torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        return ids

    def decode(self, ids):
        return ' '.join([self.id_to_token[i] for i in ids])

# Dataset class
class BurmeseDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.texts = [tokenizer.encode(text) for text in texts]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        input_ids = text[:self.seq_len]
        attention_mask = [1] * len(input_ids) + [0] * (self.seq_len - len(input_ids))
        input_ids += [self.tokenizer.token_to_id["[PAD]"]] * (self.seq_len - len(input_ids))
        return torch.tensor(input_ids), torch.tensor(attention_mask)

# Additional function for calculating perplexity
def calculate_perplexity(log_probs, targets):
    """
    Calculate perplexity from log probabilities.
    :param log_probs: Log probabilities of the predictions (Tensor)
    :param targets: Ground truth token IDs (Tensor)
    :return: Perplexity value (float)
    """
    target_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    avg_log_prob = target_log_probs.mean()
    return torch.exp(-avg_log_prob).item()

def calculate_cross_entropy(log_probs, targets):
    """
    Calculate cross-entropy loss from log probabilities.
    :param log_probs: Log probabilities of the predictions (Tensor)
    :param targets: Ground truth token IDs (Tensor)
    :return: Cross-entropy value (float)
    """
    # Flatten the log_probs and targets to compute cross-entropy
    log_probs_flat = log_probs.view(-1, log_probs.size(-1))
    targets_flat = targets.view(-1)
    
    # Compute the cross-entropy loss
    cross_entropy = torch.nn.functional.cross_entropy(log_probs_flat, targets_flat, reduction='mean')
    return cross_entropy.item()
