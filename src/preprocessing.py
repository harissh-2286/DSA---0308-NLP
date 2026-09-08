"""
NLP Preprocessing, Domain Tokenization, Abbreviation Expansion, and PyTorch Dataset Loaders.
"""

import os
import re
import json
import torch
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Comprehensive domain abbreviation dictionary for maintenance narratives
DOMAIN_ABBREVIATIONS = {
    r"\bvib\b|\bvibs\b": "vibration",
    r"\bbrg\b|\bbrgs\b": "bearing",
    r"\bde\b": "drive end",
    r"\bnde\b": "non drive end",
    r"\bpm\b": "preventative maintenance",
    r"\bcm\b": "corrective maintenance",
    r"\brtd\b|\brtds\b": "resistance temperature detector",
    r"\brpm\b": "revolutions per minute",
    r"\bpsi\b": "pounds per square inch",
    r"\bgmf\b": "gear mesh frequency",
    r"\bthd\b": "total harmonic distortion",
    r"\boem\b": "original equipment manufacturer",
    r"\bep\b": "extreme pressure",
    r"\bmohm\b|\bmohms\b": "mega ohms",
    r"\bhz\b": "hertz",
    r"\bmtr\b": "motor",
    r"\bmech\b": "mechanical",
    r"\btemp\b|\btemps\b": "temperature",
    r"\bpress\b": "pressure",
    r"\bapprox\b": "approximately",
    r"\breplaced\b": "replaced",
    r"\bo/h\b": "overhaul",
    r"\bo-ring\b|\boring\b": "o ring seal"
}

# Multi-persona domain symptom dictionary for explainability / feature tagging
# Includes formal technical terms, informal slang, acoustic descriptions, and sensory cues
DOMAIN_SYMPTOMS = {
    "Vibration / Imbalance": [
        "vibration", "vibrations", "vibrating", "shaking", "wobble", "wobbling", "tremor", "shudder",
        "unbalance", "resonance", "harmonic", "amplitude", "sideband", "looseness", "jitter"
    ],
    "Thermal / Heat": [
        "temperature", "hot", "hotspot", "overheating", "overheated", "thermal", "smoking",
        "smoke", "burning", "burnt", "scorching", "varnish", "casing temp", "warm to touch"
    ],
    "Acoustic / Noise": [
        "whining", "grinding", "rattle", "rattling", "knocking", "clunking", "clank", "squeal",
        "screeching", "screech", "whistle", "whistling", "cavitation", "humming", "pitch", "buzzing", "groan"
    ],
    "Mechanical Wear": [
        "spalling", "pitting", "scuffing", "brinelled", "backlash", "fracture", "chipped",
        "debris", "wear", "metal flakes", "shavings", "scoring", "damaged tooth", "broken",
        "crack", "cracks", "cracked", "flaking", "groove", "grooves", "seizure"
    ],
    "Fluid / Seal Leak": [
        "seepage", "seeping", "weeping", "leakage", "leak", "leaking", "dripping", "puddle",
        "fluid loss", "depressurization", "o ring", "seal", "hydraulic", "hose burst", "pressure drop",
        "oil", "fluid", "oil leak", "spill"
    ],
    "Electrical Anomaly": [
        "arcing", "sparks", "sparking", "megger", "insulation", "impedance", "flashover",
        "short circuit", "shorted", "earth fault", "tripped breaker", "blown fuse", "carbonized"
    ]
}

# Negation patterns to prevent false symptom alarms (e.g. "no vibration detected")
NEGATION_PATTERNS = [
    r"\bno\s+",
    r"\bzero\s+",
    r"\bwithout\s+",
    r"\bnot\s+",
    r"\bno\s+abnormal\s+",
    r"\bno\s+evidence\s+of\s+",
    r"\bneither\s+"
]

FAILURE_MODE_MAP = {
    "Normal_Operation": 0,
    "Bearing_Degradation": 1,
    "Motor_Overheating": 2,
    "Hydraulic_Leak": 3,
    "Gearbox_Tooth_Wear": 4,
    "Electrical_Fault": 5
}
REV_FAILURE_MODE_MAP = {v: k for k, v in FAILURE_MODE_MAP.items()}

RISK_LEVEL_MAP = {
    "Normal": 0,
    "Low_Risk": 1,
    "Medium_Risk": 2,
    "Critical": 3
}
REV_RISK_LEVEL_MAP = {v: k for k, v in RISK_LEVEL_MAP.items()}


def clean_maintenance_text(text: str, expand_abbreviations: bool = True) -> str:
    """
    Cleans raw maintenance narrative from any author (junior tech, engineer, voice dictation, shift log).
    Handles lowercase normalization, slang/abbreviation expansion, and whitespace standardization.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    
    if expand_abbreviations:
        for pattern, repl in DOMAIN_ABBREVIATIONS.items():
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
            
    # Clean up non-alphanumeric punctuation while retaining units & decimals
    text = re.sub(r"[^\w\s\.\,\-\:\/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_symptoms_and_entities(text: str) -> dict:
    """
    Extracts detected domain failure symptoms while accounting for author negation.
    (e.g., 'no vibration or leak' will NOT flag vibration or leak).
    """
    cleaned = clean_maintenance_text(text, expand_abbreviations=True)
    detected = {}
    
    # Split into clauses/sentences
    clauses = re.split(r"[\.\,\;\n]", cleaned)
    
    for category, keywords in DOMAIN_SYMPTOMS.items():
        found = set()
        for kw in keywords:
            for clause in clauses:
                clause = clause.strip()
                if not clause:
                    continue
                match = re.search(r"\b" + re.escape(kw) + r"\b", clause)
                if match:
                    # Check if preceded by negation within the clause
                    prefix = clause[:match.start()].strip()
                    # Check if prefix contains negation words like 'no', 'zero', 'without', 'not', 'no evidence of'
                    is_negated = bool(re.search(r"\b(no|zero|without|not|none|never)\b(?:\s+\w+){0,3}$", prefix))
                    if not is_negated:
                        found.add(kw)
        if found:
            detected[category] = sorted(list(found))
            
    return detected


class Vocabulary:
    """Vocabulary mapping for BiLSTM token sequences."""
    def __init__(self, pad_token="<PAD>", unk_token="<UNK>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word2idx = {pad_token: 0, unk_token: 1}
        self.idx2word = {0: pad_token, 1: unk_token}
        self.freqs = Counter()

    def build_vocab(self, texts, max_size=5000, min_freq=1):
        for text in texts:
            tokens = text.split()
            self.freqs.update(tokens)
            
        for word, count in self.freqs.most_common(max_size):
            if count >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, text, max_len=64):
        tokens = text.split()
        indices = [self.word2idx.get(t, self.word2idx[self.unk_token]) for t in tokens[:max_len]]
        # Pad sequence
        if len(indices) < max_len:
            indices = indices + [self.word2idx[self.pad_token]] * (max_len - len(indices))
        return indices

    def save(self, filepath):
        with open(filepath, "w") as f:
            json.dump({"word2idx": self.word2idx, "idx2word": {str(k): v for k, v in self.idx2word.items()}}, f, indent=2)

    @classmethod
    def load(cls, filepath):
        vocab = cls()
        with open(filepath, "r") as f:
            data = json.load(f)
            vocab.word2idx = data["word2idx"]
            vocab.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        return vocab

    def __len__(self):
        return len(self.word2idx)


class MaintenanceBiLSTMDataset(Dataset):
    """PyTorch Dataset for BiLSTM model."""
    def __init__(self, texts, failure_labels, risk_labels, early_warn_labels, vocab, max_len=64):
        self.encoded_texts = [vocab.encode(t, max_len=max_len) for t in texts]
        self.failure_labels = failure_labels
        self.risk_labels = risk_labels
        self.early_warn_labels = early_warn_labels

    def __len__(self):
        return len(self.encoded_texts)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encoded_texts[idx], dtype=torch.long),
            "failure_label": torch.tensor(self.failure_labels[idx], dtype=torch.long),
            "risk_label": torch.tensor(self.risk_labels[idx], dtype=torch.long),
            "early_warn_label": torch.tensor(self.early_warn_labels[idx], dtype=torch.float)
        }


class MaintenanceTransformerDataset(Dataset):
    """PyTorch Dataset for BERT / RoBERTa models using HuggingFace Tokenizers."""
    def __init__(self, texts, failure_labels, risk_labels, early_warn_labels, tokenizer, max_len=128):
        self.texts = list(texts)
        self.failure_labels = failure_labels
        self.risk_labels = risk_labels
        self.early_warn_labels = early_warn_labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "failure_label": torch.tensor(self.failure_labels[idx], dtype=torch.long),
            "risk_label": torch.tensor(self.risk_labels[idx], dtype=torch.long),
            "early_warn_label": torch.tensor(self.early_warn_labels[idx], dtype=torch.float)
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
        return item


def load_and_preprocess_data(csv_path="data/industrial_maintenance_dataset.csv", test_size=0.15, val_size=0.15, random_state=42):
    """
    Loads dataset, performs stratified split by equipment_id to prevent temporal leakage,
    and returns processed dataframes & label encoders.
    """
    df = pd.read_csv(csv_path)
    df["cleaned_narrative"] = df["narrative"].apply(clean_maintenance_text)
    
    # Stratified split based on unique equipment IDs and their final failure mode
    eq_groups = df.groupby("equipment_id").agg({
        "failure_mode": "last",
        "risk_level": "last"
    }).reset_index()
    
    train_eqs, temp_eqs = train_test_split(
        eq_groups["equipment_id"].values,
        test_size=(test_size + val_size),
        stratify=eq_groups["failure_mode"].values,
        random_state=random_state
    )
    
    temp_df = eq_groups[eq_groups["equipment_id"].isin(temp_eqs)]
    val_ratio = val_size / (test_size + val_size)
    val_eqs, test_eqs = train_test_split(
        temp_df["equipment_id"].values,
        test_size=(1.0 - val_ratio),
        stratify=temp_df["failure_mode"].values,
        random_state=random_state
    )
    
    train_df = df[df["equipment_id"].isin(train_eqs)].reset_index(drop=True)
    val_df = df[df["equipment_id"].isin(val_eqs)].reset_index(drop=True)
    test_df = df[df["equipment_id"].isin(test_eqs)].reset_index(drop=True)
    
    print(f"Data Split Summary: Train={len(train_df)} ({len(train_eqs)} machines), Val={len(val_df)} ({len(val_eqs)} machines), Test={len(test_df)} ({len(test_eqs)} machines)")
    
    return train_df, val_df, test_df
