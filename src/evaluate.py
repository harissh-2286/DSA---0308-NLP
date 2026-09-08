import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
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
    REV_FAILURE_MODE_MAP,
    RISK_LEVEL_MAP,
    REV_RISK_LEVEL_MAP
)
from src.models import (
    BiLSTMAttentionClassifier,
    MaintenanceBERTClassifier,
    MaintenanceRoBERTaClassifier,
    TemporalDegradationLSTM
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_confusion_matrix(cm, class_names, title, save_filename):
    """Generates and saves a high-quality styled confusion matrix chart."""
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="white")
    
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5,
        linecolor="#dddddd"
    )
    plt.title(title, fontsize=14, pad=15, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.ylabel("Ground Truth Class", fontsize=11, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {save_path}")


def evaluate_bilstm(test_df, vocab):
    print("\n--- Evaluating BiLSTM with Attention on Test Set ---")
    model_path = os.path.join(OUTPUT_DIR, "best_bilstm_model.pt")
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return None
        
    model = BiLSTMAttentionClassifier(
        vocab_size=len(vocab),
        embed_dim=128,
        hidden_dim=128,
        num_layers=2,
        num_failure_classes=len(FAILURE_MODE_MAP),
        num_risk_classes=len(RISK_LEVEL_MAP)
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    test_ds = MaintenanceBiLSTMDataset(
        test_df["cleaned_narrative"].values,
        test_df["failure_mode"].map(FAILURE_MODE_MAP).values,
        test_df["risk_level"].map(RISK_LEVEL_MAP).values,
        test_df["early_warning_flag"].values,
        vocab=vocab,
        max_len=64
    )
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    y_true_f, y_pred_f = [], []
    y_true_r, y_pred_r = [], []
    y_true_e, y_pred_e = [], []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            out = model(input_ids)
            
            y_pred_f.extend(out["failure_logits"].argmax(dim=-1).cpu().numpy())
            y_true_f.extend(batch["failure_label"].numpy())
            
            y_pred_r.extend(out["risk_logits"].argmax(dim=-1).cpu().numpy())
            y_true_r.extend(batch["risk_label"].numpy())
            
            pred_early = (torch.sigmoid(out["early_warn_logits"]) >= 0.5).cpu().numpy().astype(int)
            y_pred_e.extend(pred_early)
            y_true_e.extend(batch["early_warn_label"].numpy().astype(int))
            
    acc_f = accuracy_score(y_true_f, y_pred_f)
    p_f, r_f, f1_f, _ = precision_recall_fscore_support(y_true_f, y_pred_f, average="macro")
    acc_r = accuracy_score(y_true_r, y_pred_r)
    p_r, r_r, f1_r, _ = precision_recall_fscore_support(y_true_r, y_pred_r, average="macro")
    acc_e = accuracy_score(y_true_e, y_pred_e)
    
    # Confusion Matrix
    cm_f = confusion_matrix(y_true_f, y_pred_f)
    failure_names = [REV_FAILURE_MODE_MAP[i] for i in range(len(FAILURE_MODE_MAP))]
    plot_confusion_matrix(cm_f, failure_names, "BiLSTM - Failure Mode Confusion Matrix", "cm_bilstm_failure.png")
    
    return {
        "model": "BiLSTM + Attention",
        "failure_accuracy": acc_f,
        "failure_macro_f1": f1_f,
        "failure_precision": p_f,
        "failure_recall": r_f,
        "risk_accuracy": acc_r,
        "risk_macro_f1": f1_r,
        "early_warning_accuracy": acc_e
    }


def evaluate_transformer(model_type, model_class, pretrained_name, test_df):
    print(f"\n--- Evaluating {model_type.upper()} ({pretrained_name}) on Test Set ---")
    model_path = os.path.join(OUTPUT_DIR, f"best_{model_type}_model.pt")
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return None
        
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    model = model_class(
        pretrained_model_name=pretrained_name,
        num_failure_classes=len(FAILURE_MODE_MAP),
        num_risk_classes=len(RISK_LEVEL_MAP)
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    test_ds = MaintenanceTransformerDataset(
        test_df["cleaned_narrative"].values,
        test_df["failure_mode"].map(FAILURE_MODE_MAP).values,
        test_df["risk_level"].map(RISK_LEVEL_MAP).values,
        test_df["early_warning_flag"].values,
        tokenizer=tokenizer,
        max_len=96
    )
    loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    y_true_f, y_pred_f = [], []
    y_true_r, y_pred_r = [], []
    y_true_e, y_pred_e = [], []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            
            if model_type == "bert":
                out = model(input_ids, attention_mask=attn_mask)
            else:
                out = model(input_ids, attention_mask=attn_mask)
                
            y_pred_f.extend(out["failure_logits"].argmax(dim=-1).cpu().numpy())
            y_true_f.extend(batch["failure_label"].numpy())
            
            y_pred_r.extend(out["risk_logits"].argmax(dim=-1).cpu().numpy())
            y_true_r.extend(batch["risk_label"].numpy())
            
            pred_early = (torch.sigmoid(out["early_warn_logits"]) >= 0.5).cpu().numpy().astype(int)
            y_pred_e.extend(pred_early)
            y_true_e.extend(batch["early_warn_label"].numpy().astype(int))
            
    acc_f = accuracy_score(y_true_f, y_pred_f)
    p_f, r_f, f1_f, _ = precision_recall_fscore_support(y_true_f, y_pred_f, average="macro")
    acc_r = accuracy_score(y_true_r, y_pred_r)
    p_r, r_r, f1_r, _ = precision_recall_fscore_support(y_true_r, y_pred_r, average="macro")
    acc_e = accuracy_score(y_true_e, y_pred_e)
    
    # Plot Confusion Matrix
    cm_f = confusion_matrix(y_true_f, y_pred_f)
    failure_names = [REV_FAILURE_MODE_MAP[i] for i in range(len(FAILURE_MODE_MAP))]
    plot_confusion_matrix(cm_f, failure_names, f"{model_type.upper()} - Failure Mode Confusion Matrix", f"cm_{model_type}_failure.png")
    
    return {
        "model": model_type.upper(),
        "failure_accuracy": acc_f,
        "failure_macro_f1": f1_f,
        "failure_precision": p_f,
        "failure_recall": r_f,
        "risk_accuracy": acc_r,
        "risk_macro_f1": f1_r,
        "early_warning_accuracy": acc_e
    }


def generate_benchmark_comparison():
    """Evaluates all trained models and generates a benchmark table and comparative bar chart."""
    _, _, test_df = load_and_preprocess_data()
    vocab = Vocabulary.load("data/vocabulary.json")
    
    results = []
    
    # 1. BiLSTM
    res_bilstm = evaluate_bilstm(test_df, vocab)
    if res_bilstm:
        results.append(res_bilstm)
        
    # 2. BERT
    res_bert = evaluate_transformer("bert", MaintenanceBERTClassifier, "bert-base-uncased", test_df)
    if res_bert:
        results.append(res_bert)
        
    # 3. RoBERTa
    res_roberta = evaluate_transformer("roberta", MaintenanceRoBERTaClassifier, "roberta-base", test_df)
    if res_roberta:
        results.append(res_roberta)
        
    benchmark_df = pd.DataFrame(results)
    
    # Format and save summary table
    table_path = os.path.join(OUTPUT_DIR, "model_benchmark_results.csv")
    benchmark_df.to_csv(table_path, index=False)
    
    print("\n" + "="*80)
    print(" NLP CAPSTONE MODEL PERFORMANCE COMPARISON BENCHMARK")
    print("="*80)
    print(benchmark_df.to_string(index=False))
    print("="*80)
    
    # Plot Comparative Bar Chart
    if len(benchmark_df) > 0:
        plt.figure(figsize=(10, 6))
        x = np.arange(len(benchmark_df))
        width = 0.22
        
        plt.bar(x - width, benchmark_df["failure_accuracy"] * 100, width, label="Failure Acc (%)", color="#3b82f6")
        plt.bar(x, benchmark_df["failure_macro_f1"] * 100, width, label="Failure Macro F1 (%)", color="#10b981")
        plt.bar(x + width, benchmark_df["early_warning_accuracy"] * 100, width, label="Early Warning Acc (%)", color="#f59e0b")
        
        plt.xlabel("Model Architecture", fontweight="bold", fontsize=12)
        plt.ylabel("Performance Score (%)", fontweight="bold", fontsize=12)
        plt.title("NLP Architecture Benchmark: Maintenance Narrative Failure Prediction", fontweight="bold", fontsize=14, pad=15)
        plt.xticks(x, benchmark_df["model"], fontweight="bold")
        plt.ylim(0, 105)
        plt.legend(frameon=True, facecolor="#ffffff", edgecolor="#cccccc")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        
        chart_path = os.path.join(OUTPUT_DIR, "model_comparison_benchmark.png")
        plt.savefig(chart_path, dpi=300)
        plt.close()
        print(f"Saved model comparison chart to: {chart_path}")
        
    return benchmark_df


if __name__ == "__main__":
    generate_benchmark_comparison()
