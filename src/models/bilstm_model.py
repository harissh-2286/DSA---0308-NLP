"""
Bidirectional LSTM with Self-Attention for Industrial Maintenance Narrative Classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Computes additive self-attention over LSTM hidden states to capture critical failure symptoms.
    """
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_output, mask=None):
        # lstm_output: (batch_size, seq_len, hidden_dim)
        energy = self.projection(lstm_output).squeeze(2)  # (batch_size, seq_len)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e9)
            
        weights = F.softmax(energy, dim=1)  # (batch_size, seq_len)
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)  # (batch_size, hidden_dim)
        return context, weights


class BiLSTMAttentionClassifier(nn.Module):
    """
    Deep BiLSTM with Attention & Multi-Task heads for Failure Mode & Risk Tier Prediction.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_failure_classes: int = 6,
        num_risk_classes: int = 4,
        dropout: float = 0.3,
        pad_idx: int = 0
    ):
        super(BiLSTMAttentionClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # BiLSTM hidden size is 2 * hidden_dim
        self.lstm_out_dim = hidden_dim * 2
        self.attention = SelfAttention(self.lstm_out_dim)
        
        self.shared_fc = nn.Sequential(
            nn.Linear(self.lstm_out_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Task Heads
        self.failure_head = nn.Linear(128, num_failure_classes)
        self.risk_head = nn.Linear(128, num_risk_classes)
        self.early_warn_head = nn.Linear(128, 1)

    def forward(self, input_ids, mask=None):
        # input_ids: (batch_size, seq_len)
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embed_dim)
        
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, 2 * hidden_dim)
        context, attn_weights = self.attention(lstm_out, mask)
        
        shared_rep = self.shared_fc(context)
        
        failure_logits = self.failure_head(shared_rep)
        risk_logits = self.risk_head(shared_rep)
        early_warn_logits = self.early_warn_head(shared_rep).squeeze(-1)
        
        return {
            "failure_logits": failure_logits,
            "risk_logits": risk_logits,
            "early_warn_logits": early_warn_logits,
            "attention_weights": attn_weights,
            "embedding_repr": context
        }
