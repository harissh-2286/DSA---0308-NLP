"""
RoBERTa-based Classifier for Industrial Equipment Failure Prediction.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class MaintenanceRoBERTaClassifier(nn.Module):
    """
    RoBERTa Architecture for Robust Industrial Maintenance Narrative Risk Assessment.
    """
    def __init__(
        self,
        pretrained_model_name: str = "roberta-base",
        num_failure_classes: int = 6,
        num_risk_classes: int = 4,
        dropout: float = 0.2
    ):
        super(MaintenanceRoBERTaClassifier, self).__init__()
        
        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        self.roberta = AutoModel.from_pretrained(pretrained_model_name)
        hidden_size = self.config.hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, 256)
        self.activation = nn.GELU()
        
        # Classification Heads
        self.failure_head = nn.Linear(256, num_failure_classes)
        self.risk_head = nn.Linear(256, num_risk_classes)
        self.early_warn_head = nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        
        # RoBERTa [<s>] token representation
        cls_repr = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_repr)
        x = self.activation(self.dense(x))
        
        failure_logits = self.failure_head(x)
        risk_logits = self.risk_head(x)
        early_warn_logits = self.early_warn_head(x).squeeze(-1)
        
        return {
            "failure_logits": failure_logits,
            "risk_logits": risk_logits,
            "early_warn_logits": early_warn_logits,
            "embedding_repr": cls_repr
        }
