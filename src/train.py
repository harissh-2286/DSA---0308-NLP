import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.preprocessing import (
    load_and_preprocess_data,
    Vocabulary,
    MaintenanceBiLSTMDataset,
    MaintenanceTransformerDataset,
    FAILURE_MODE_MAP,
    RISK_LEVEL_MAP
)
from src.models import (
    BiLSTMAttentionClassifier,
    MaintenanceBERTClassifier,
    MaintenanceRoBERTaClassifier,
    TemporalDegradationLSTM
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def train_bilstm(train_df, val_df, vocab, epochs=15, batch_size=32, lr=1e-3):
    print("\n" + "="*60)
    print(f" TRAINING BiLSTM ATTENTION MODEL ON {DEVICE}")
    print("="*60)
    
    train_ds = MaintenanceBiLSTMDataset(
        train_df["cleaned_narrative"].values,
        train_df["failure_mode"].map(FAILURE_MODE_MAP).values,
        train_df["risk_level"].map(RISK_LEVEL_MAP).values,
        train_df["early_warning_flag"].values,
        vocab=vocab,
        max_len=64
    )
    val_ds = MaintenanceBiLSTMDataset(
        val_df["cleaned_narrative"].values,
        val_df["failure_mode"].map(FAILURE_MODE_MAP).values,
        val_df["risk_level"].map(RISK_LEVEL_MAP).values,
        val_df["early_warning_flag"].values,
        vocab=vocab,
        max_len=64
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    model = BiLSTMAttentionClassifier(
        vocab_size=len(vocab),
        embed_dim=128,
        hidden_dim=128,
        num_layers=2,
        num_failure_classes=len(FAILURE_MODE_MAP),
        num_risk_classes=len(RISK_LEVEL_MAP),
        dropout=0.3
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion_failure = nn.CrossEntropyLoss()
    criterion_risk = nn.CrossEntropyLoss()
    criterion_early = nn.BCEWithLogitsLoss()
    
    best_val_acc = 0.0
    history = {"train_loss": [], "val_acc_failure": [], "val_acc_risk": []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            y_fail = batch["failure_label"].to(DEVICE)
            y_risk = batch["risk_label"].to(DEVICE)
            y_early = batch["early_warn_label"].to(DEVICE)
            
            optimizer.zero_grad()
            out = model(input_ids)
            
            loss_fail = criterion_failure(out["failure_logits"], y_fail)
            loss_risk = criterion_risk(out["risk_logits"], y_risk)
            loss_early = criterion_early(out["early_warn_logits"], y_early)
            
            loss = loss_fail + 0.8 * loss_risk + 0.5 * loss_early
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        correct_fail, correct_risk, total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                y_fail = batch["failure_label"].to(DEVICE)
                y_risk = batch["risk_label"].to(DEVICE)
                
                out = model(input_ids)
                pred_fail = out["failure_logits"].argmax(dim=-1)
                pred_risk = out["risk_logits"].argmax(dim=-1)
                
                correct_fail += (pred_fail == y_fail).sum().item()
                correct_risk += (pred_risk == y_risk).sum().item()
                total += len(y_fail)
                
        val_acc_f = correct_fail / total
        val_acc_r = correct_risk / total
        avg_loss = total_loss / len(train_loader)
        
        history["train_loss"].append(avg_loss)
        history["val_acc_failure"].append(val_acc_f)
        history["val_acc_risk"].append(val_acc_r)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_loss:.4f} | Val Fail Acc: {val_acc_f*100:.2f}% | Val Risk Acc: {val_acc_r*100:.2f}%")
        
        if val_acc_f > best_val_acc:
            best_val_acc = val_acc_f
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_bilstm_model.pt"))
            
    print(f"--> BiLSTM Training Complete. Best Validation Failure Accuracy: {best_val_acc*100:.2f}%")
    return model, history


def train_transformer_model(model_type, model_class, pretrained_name, train_df, val_df, epochs=5, batch_size=16, lr=2e-5):
    print("\n" + "="*60)
    print(f" TRAINING {model_type.upper()} ({pretrained_name}) ON {DEVICE}")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    
    train_ds = MaintenanceTransformerDataset(
        train_df["cleaned_narrative"].values,
        train_df["failure_mode"].map(FAILURE_MODE_MAP).values,
        train_df["risk_level"].map(RISK_LEVEL_MAP).values,
        train_df["early_warning_flag"].values,
        tokenizer=tokenizer,
        max_len=96
    )
    val_ds = MaintenanceTransformerDataset(
        val_df["cleaned_narrative"].values,
        val_df["failure_mode"].map(FAILURE_MODE_MAP).values,
        val_df["risk_level"].map(RISK_LEVEL_MAP).values,
        val_df["early_warning_flag"].values,
        tokenizer=tokenizer,
        max_len=96
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    model = model_class(
        pretrained_model_name=pretrained_name,
        num_failure_classes=len(FAILURE_MODE_MAP),
        num_risk_classes=len(RISK_LEVEL_MAP),
        dropout=0.2
    ).to(DEVICE)
    
    # Differential learning rates: smaller for transformer backbone, larger for custom heads
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "head" not in n and "dense" not in n], "lr": lr},
        {"params": [p for n, p in model.named_parameters() if "head" in n or "dense" in n], "lr": lr * 5}
    ], weight_decay=0.01)
    
    criterion_failure = nn.CrossEntropyLoss()
    criterion_risk = nn.CrossEntropyLoss()
    criterion_early = nn.BCEWithLogitsLoss()
    
    best_val_acc = 0.0
    history = {"train_loss": [], "val_acc_failure": [], "val_acc_risk": []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            y_fail = batch["failure_label"].to(DEVICE)
            y_risk = batch["risk_label"].to(DEVICE)
            y_early = batch["early_warn_label"].to(DEVICE)
            
            optimizer.zero_grad()
            out = model(input_ids, attention_mask=attn_mask)
            
            loss_fail = criterion_failure(out["failure_logits"], y_fail)
            loss_risk = criterion_risk(out["risk_logits"], y_risk)
            loss_early = criterion_early(out["early_warn_logits"], y_early)
            
            loss = loss_fail + 0.8 * loss_risk + 0.5 * loss_early
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        correct_fail, correct_risk, total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attn_mask = batch["attention_mask"].to(DEVICE)
                y_fail = batch["failure_label"].to(DEVICE)
                y_risk = batch["risk_label"].to(DEVICE)
                
                out = model(input_ids, attention_mask=attn_mask)
                pred_fail = out["failure_logits"].argmax(dim=-1)
                pred_risk = out["risk_logits"].argmax(dim=-1)
                
                correct_fail += (pred_fail == y_fail).sum().item()
                correct_risk += (pred_risk == y_risk).sum().item()
                total += len(y_fail)
                
        val_acc_f = correct_fail / total
        val_acc_r = correct_risk / total
        avg_loss = total_loss / len(train_loader)
        
        history["train_loss"].append(avg_loss)
        history["val_acc_failure"].append(val_acc_f)
        history["val_acc_risk"].append(val_acc_r)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_loss:.4f} | Val Fail Acc: {val_acc_f*100:.2f}% | Val Risk Acc: {val_acc_r*100:.2f}%")
        
        if val_acc_f > best_val_acc:
            best_val_acc = val_acc_f
            save_path = os.path.join(OUTPUT_DIR, f"best_{model_type}_model.pt")
            torch.save(model.state_dict(), save_path)
            
    print(f"--> {model_type.upper()} Training Complete. Best Validation Failure Accuracy: {best_val_acc*100:.2f}%")
    return model, history


def train_temporal_model(bilstm_feature_extractor, vocab, train_df, val_df, epochs=12, batch_size=16, lr=1e-3):
    print("\n" + "="*60)
    print(f" TRAINING TEMPORAL DEGRADATION BI-LSTM ON {DEVICE}")
    print("="*60)
    
    bilstm_feature_extractor.eval()
    
    def prepare_temporal_sequences(df):
        seq_list = []
        labels_fail = []
        labels_risk = []
        labels_early = []
        labels_ttf = []
        
        grouped = df.groupby("equipment_id")
        for eq_id, group in grouped:
            group_sorted = group.sort_values("visit_sequence")
            narratives = group_sorted["cleaned_narrative"].values
            
            # Extract BiLSTM text representations
            encoded = [vocab.encode(t, max_len=64) for t in narratives]
            input_tensor = torch.tensor(encoded, dtype=torch.long).to(DEVICE)
            
            with torch.no_grad():
                out = bilstm_feature_extractor(input_tensor)
                text_embs = out["embedding_repr"].cpu().numpy()  # (num_visits, 128)
                
            # Telemetry features normalized: vib/10, temp/150, press/250, rpm/6000
            vib = group_sorted["vibration_mms"].values[:, None] / 10.0
            temp = group_sorted["temperature_c"].values[:, None] / 150.0
            press = group_sorted["pressure_psi"].values[:, None] / 250.0
            rpm = group_sorted["rpm"].values[:, None] / 6000.0
            
            telemetry = np.hstack([vib, temp, press, rpm])
            combined_features = np.hstack([text_embs, telemetry]) # (num_visits, 132)
            
            seq_list.append(combined_features)
            labels_fail.append(FAILURE_MODE_MAP[group_sorted["failure_mode"].iloc[-1]])
            labels_risk.append(RISK_LEVEL_MAP[group_sorted["risk_level"].iloc[-1]])
            labels_early.append(float(group_sorted["early_warning_flag"].iloc[-1]))
            labels_ttf.append(float(group_sorted["time_to_failure_days"].iloc[-1]))
            
        # Pad sequences to max length
        max_seq = max(len(s) for s in seq_list)
        feat_dim = seq_list[0].shape[1]
        
        padded_seqs = np.zeros((len(seq_list), max_seq, feat_dim), dtype=np.float32)
        for i, s in enumerate(seq_list):
            padded_seqs[i, :len(s), :] = s
            
        return (
            torch.tensor(padded_seqs, dtype=torch.float32),
            torch.tensor(labels_fail, dtype=torch.long),
            torch.tensor(labels_risk, dtype=torch.long),
            torch.tensor(labels_early, dtype=torch.float32),
            torch.tensor(labels_ttf, dtype=torch.float32)
        )
        
    X_train, y_f_train, y_r_train, y_e_train, y_ttf_train = prepare_temporal_sequences(train_df)
    X_val, y_f_val, y_r_val, y_e_val, y_ttf_val = prepare_temporal_sequences(val_df)
    
    temporal_model = TemporalDegradationLSTM(
        feature_dim=X_train.shape[2],
        hidden_dim=64,
        num_layers=2,
        num_failure_classes=len(FAILURE_MODE_MAP),
        num_risk_classes=len(RISK_LEVEL_MAP),
        dropout=0.2
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(temporal_model.parameters(), lr=lr, weight_decay=1e-4)
    crit_fail = nn.CrossEntropyLoss()
    crit_risk = nn.CrossEntropyLoss()
    crit_early = nn.BCEWithLogitsLoss()
    crit_ttf = nn.MSELoss()
    
    num_samples = len(X_train)
    best_val_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        temporal_model.train()
        perm = torch.randperm(num_samples)
        total_loss = 0.0
        batches = 0
        
        for i in range(0, num_samples, batch_size):
            idx = perm[i:i + batch_size]
            batch_x = X_train[idx].to(DEVICE)
            batch_yf = y_f_train[idx].to(DEVICE)
            batch_yr = y_r_train[idx].to(DEVICE)
            batch_ye = y_e_train[idx].to(DEVICE)
            batch_yttf = y_ttf_train[idx].to(DEVICE)
            
            optimizer.zero_grad()
            out = temporal_model(batch_x)
            
            l_fail = crit_fail(out["failure_logits"], batch_yf)
            l_risk = crit_risk(out["risk_logits"], batch_yr)
            l_early = crit_early(out["early_warn_logits"], batch_ye)
            l_ttf = crit_ttf(out["ttf_pred"], batch_yttf) * 0.001
            
            loss = l_fail + 0.8 * l_risk + 0.5 * l_early + l_ttf
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
            
        # Validation
        temporal_model.eval()
        with torch.no_grad():
            out_val = temporal_model(X_val.to(DEVICE))
            pred_f = out_val["failure_logits"].argmax(dim=-1).cpu()
            val_acc = (pred_f == y_f_val).float().mean().item()
            
        print(f"Epoch {epoch:02d}/{epochs:02d} | Temporal Train Loss: {total_loss/batches:.4f} | Val Failure Acc: {val_acc*100:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(temporal_model.state_dict(), os.path.join(OUTPUT_DIR, "best_temporal_lstm_model.pt"))
            
    print(f"--> Temporal Model Training Complete. Best Validation Acc: {best_val_acc*100:.2f}%")
    return temporal_model


def run_full_training():
    start_time = time.time()
    train_df, val_df, test_df = load_and_preprocess_data()
    
    # 1. Build and save Vocabulary for BiLSTM
    vocab = Vocabulary()
    vocab.build_vocab(train_df["cleaned_narrative"].values, max_size=4000, min_freq=1)
    vocab_path = os.path.join("data", "vocabulary.json")
    vocab.save(vocab_path)
    print(f"Built and saved vocabulary ({len(vocab)} words) to: {vocab_path}")
    
    # 2. Train BiLSTM with Attention
    bilstm_model, _ = train_bilstm(train_df, val_df, vocab, epochs=12, batch_size=32, lr=1e-3)
    
    # 3. Train Temporal Degradation LSTM
    _ = train_temporal_model(bilstm_model, vocab, train_df, val_df, epochs=10, batch_size=16)
    
    # 4. Train BERT Model
    try:
        train_sub = train_df.sample(n=min(600, len(train_df)), random_state=42)
        val_sub = val_df.sample(n=min(200, len(val_df)), random_state=42)
        _ = train_transformer_model("bert", MaintenanceBERTClassifier, "bert-base-uncased", train_sub, val_sub, epochs=2, batch_size=16, lr=2e-5)
    except Exception as e:
        print(f"[Warning] BERT training encountered error or network timeout: {e}. Continuing with trained BiLSTM & Temporal.")
        
    # 5. Train RoBERTa Model
    try:
        train_sub = train_df.sample(n=min(600, len(train_df)), random_state=42)
        val_sub = val_df.sample(n=min(200, len(val_df)), random_state=42)
        _ = train_transformer_model("roberta", MaintenanceRoBERTaClassifier, "roberta-base", train_sub, val_sub, epochs=2, batch_size=16, lr=2e-5)
    except Exception as e:
        print(f"[Warning] RoBERTa training encountered error or network timeout: {e}. Continuing with trained BiLSTM & Temporal.")
        
    total_min = (time.time() - start_time) / 60.0
    print("\n" + "="*60)
    print(f" ALL MODEL TRAINING STAGES FINISHED SUCCESSFULLY in {total_min:.1f} mins!")
    print("="*60)


if __name__ == "__main__":
    run_full_training()
