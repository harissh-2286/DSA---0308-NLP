"""
BERT-based Classifier for Industrial Maintenance Narratives.
Uses HuggingFace BERT with custom Multi-Task Classification Heads.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class MaintenanceBERTClassifier(nn.Module):
    """
    Fine-tunable BERT Architecture for Maintenance Narrative Failure & Risk Classification.
    """
    def __init__(
        self,
        pretrained_model_name: str = "bert-base-uncased",
        num_failure_classes: int = 6,
        num_risk_classes: int = 4,
        dropout: float = 0.2
    ):
        super(MaintenanceBERTClassifier, self).__init__()
        
        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        hidden_size = self.config.hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, 256)
        self.activation = nn.GELU()
        
        # Classification Heads
        self.failure_head = nn.Linear(256, num_failure_classes)
        self.risk_head = nn.Linear(256, num_risk_classes)
        self.early_warn_head = nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
            
        outputs = self.bert(**kwargs)
        
        # Use pooled [CLS] token representation or mean pooling
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            cls_repr = outputs.pooler_output
        else:
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
