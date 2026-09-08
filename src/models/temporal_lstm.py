"""
Temporal Degradation BiLSTM for Multi-Visit Equipment Narrative & Telemetry Trajectories.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalDegradationLSTM(nn.Module):
    """
    Models multi-visit historical maintenance trajectory (narrative embeddings + telemetry features).
    """
    def __init__(
        self,
        feature_dim: int = 128 + 4, # 128 dim text embedding + 4 telemetry signals (vib, temp, press, rpm)
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_failure_classes: int = 6,
        num_risk_classes: int = 4,
        dropout: float = 0.2
    ):
        super(TemporalDegradationLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            feature_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.temporal_attn = nn.Linear(hidden_dim * 2, 1)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.failure_head = nn.Linear(64, num_failure_classes)
        self.risk_head = nn.Linear(64, num_risk_classes)
        self.early_warn_head = nn.Linear(64, 1)
        self.ttf_head = nn.Linear(64, 1) # Predicts days to failure

    def forward(self, seq_features, seq_lens=None):
        """
        seq_features: (batch_size, max_seq_len, feature_dim)
        """
        lstm_out, _ = self.lstm(seq_features) # (batch, seq_len, 2 * hidden_dim)
        
        # Temporal attention over visits
        attn_scores = self.temporal_attn(lstm_out) # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # Weighted context across visits
        context = torch.sum(attn_weights * lstm_out, dim=1) # (batch, 2 * hidden_dim)
        rep = self.fc(context)
        
        failure_logits = self.failure_head(rep)
        risk_logits = self.risk_head(rep)
        early_warn_logits = self.early_warn_head(rep).squeeze(-1)
        ttf_pred = F.relu(self.ttf_head(rep)).squeeze(-1) # TTF cannot be negative
        
        return {
            "failure_logits": failure_logits,
            "risk_logits": risk_logits,
            "early_warn_logits": early_warn_logits,
            "ttf_pred": ttf_pred,
            "temporal_attention": attn_weights.squeeze(-1)
        }
