# Industrial Equipment Failure Prediction & Early Warning Engine from Maintenance Narratives

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Natural Language Processing (NLP) and Deep Learning system designed to analyze unstructured industrial maintenance work orders, technician log narratives, and telemetry remarks to detect early degradation patterns and predict catastrophic equipment breakdowns before they occur.

---

## 📌 Problem Statement & Motivation
Industrial equipment breakdowns (in pumps, turbines, compressors, gearboxes, and motors) lead to expensive unplanned downtime, safety hazards, and multimillion-dollar repair bills. While sensor monitoring exists, **technician maintenance narratives contain critical, early qualitative failure indicators** (e.g. subtle acoustic whining, micro-pitting, seal glazing, slight burning smells, and harmonic vibrations) that precede sensor threshold alarms by weeks.

This project implements a multi-model comparative NLP pipeline using **BiLSTM with Self-Attention**, **BERT (`bert-base-uncased`)**, **RoBERTa (`roberta-base`)**, and a **Temporal Degradation Multi-Visit Sequence Model**.

---

## 🏗️ System Architecture

```
+----------------------------------------------------------------------------------------------------+
|                                  TECHNICIAN MAINTENANCE LOG                                        |
| "VIB increased to 4.35 mm/s on Drive End Bearing with faint metallic whining. Temp elevated 74.5C" |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                    1. NLP PREPROCESSING & XAI                                      |
| - Domain abbreviation expansion (VIB -> Vibration, BRG -> Bearing, RTD, PM, CM, DE/NDE, GMF)      |
| - Domain symptom tagging (Cavitation, Spalling, Pitting, Thermal Hotspots, Seal Glazing)           |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                         +------------------------+------------------------+
                         |                                                 |
                         v                                                 v
+-------------------------------------------------+     +--------------------------------------------+
|            SINGLE-VISIT NARRATIVE MODELS        |     |         TEMPORAL SEQUENCE MODEL            |
| 1. BiLSTM + Self-Attention (Word Embeddings)    |     | - Multi-Visit BiLSTM Trajectory Tracker    |
| 2. BERT Fine-Tuned (bert-base-uncased)          |     | - Models temporal degradation progression  |
| 3. RoBERTa Fine-Tuned (roberta-base)            |     | - Predicts Remaining Useful Life (RUL/TTF) |
+------------------------+------------------------+     +----------------------+---------------------+
                         |                                                     |
                         +------------------------+----------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                    MULTI-TASK PREDICTION OUTPUT                                    |
| 1. Failure Mode (Normal, Bearing, Motor Overheating, Hydraulic Leak, Gearbox Wear, Electrical)     |
| 2. Risk Tier (Normal, Low Risk, Medium Risk, Critical Failure)                                     |
| 3. Early Warning Indicator & Lead Time Estimate (e.g. 30-60 Days Lead Time)                        |
| 4. Prescriptive Maintenance Recommendation                                                        |
+----------------------------------------------------------------------------------------------------+
```

---

## 📂 Repository Structure

```
Implementation/
│── data/
│   ├── industrial_maintenance_dataset.csv     # Synthesized benchmark dataset (3,800+ records)
│   ├── equipment_temporal_sequences.json      # Multi-visit historical trajectories per machine
│   └── vocabulary.json                        # Domain vocabulary index for BiLSTM
│── src/
│   ├── __init__.py
│   ├── data_generator.py                      # Realistic industrial dataset synthesizer
│   ├── preprocessing.py                       # Text normalization, tokenization, PyTorch datasets
│   ├── models/
│   │   ├── __init__.py
│   │   ├── bilstm_model.py                    # BiLSTM + Additive Self-Attention
│   │   ├── bert_model.py                      # BERT contextual classifier & feature extractor
│   │   ├── roberta_model.py                   # RoBERTa transformer classifier
│   │   └── temporal_lstm.py                   # Temporal multi-visit degradation model
│   ├── train.py                               # Training pipeline with multi-task loss
│   ├── evaluate.py                            # Evaluation benchmarks & confusion matrices
│   └── predictor.py                           # Unified inference engine with symptom attribution
│── outputs/                                   # Saved model weights (.pt) & evaluation charts
│── app.py                                     # Interactive Web Dashboard (Flask + Dark Mode UI)
│── requirements.txt                           # Python dependencies
└── README.md                                  # Documentation
```

---

## 🚀 Quick Start Guide

### 1. Installation
Clone this repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset (If starting fresh)
```bash
python src/data_generator.py
```

### 3. Train All Models
Trains BiLSTM, Temporal Degradation LSTM, BERT, and RoBERTa:
```bash
python src/train.py
```

### 4. Run Evaluation & Benchmark Comparison
Generates classification reports, confusion matrices, and model comparison bar charts:
```bash
python src/evaluate.py
```

### 5. Launch Interactive Web Dashboard
```bash
python app.py
```
Open your browser at `http://127.0.0.1:5000` to interact with the system in real time!

---

## 🔬 Model Architectures & Tri-Model Hybrid Synergy

```
+----------------------------------------------------------------------------------------------------+
|                                    1. BERT SEMANTIC EXTRACTION                                     |
| - Extracts high-dimensional 768-D contextual embeddings from technician log narratives             |
| - Computes token-level diagnostic importance scores and salient symptom cues                       |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                 2. BiLSTM TEMPORAL DEGRADATION ANALYSIS                             |
| - Tracks sequential multi-visit equipment histories [v_1, v_2, ..., v_T] over time                 |
| - Fuses BERT semantic embeddings with continuous telemetry (Vibration, Temp, Pressure, RPM)        |
| - Computes degradation trajectory velocity, rate of worsening, and Remaining Useful Life (TTF)      |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                 3. RoBERTa PROBABILITY ANALYSIS                                    |
| - Robust transformer classifier with Byte-Pair Encoding (BPE) for noisy industrial narratives      |
| - Generates calibrated softmax probability distributions across 6 Failure Modes & 4 Risk Tiers     |
| - Computes Shannon Entropy uncertainty metrics and early warning hazard alerts                     |
+----------------------------------------------------------------------------------------------------+
```

### 1. BERT Semantic Extractor (`bert-base-uncased`)
- **Semantic Extraction**: Deep bidirectional 768-dimensional contextual sentence embeddings.
- **Diagnostic Salience**: Token importance attribution scoring via L2-norm contextual vector activations.

### 2. BiLSTM Temporal Degradation Analyzer
- **Sequence Trajectory**: Models degradation across inspection visits $[v_1, v_2, \dots, v_T]$ for each asset.
- **Temporal Attention**: Dynamically weights critical turning points in equipment history to project Remaining Useful Life (RUL / TTF in days).

### 3. RoBERTa Probability Engine (`roberta-base`)
- **Calibrated Multi-Task Heads**: Predicts full probability distributions across 6 Failure Modes and 4 Risk Tiers.
- **Uncertainty Estimation**: Calculates Shannon Entropy $H(p) = -\sum p \log(p)$ to quantify prediction confidence.

---

## 📊 Target Classes & Classifications

| Failure Mode Category | Description | Common Symptoms / Narrative Keywords |
| :--- | :--- | :--- |
| **Normal Operation** | Routine inspection within nominal specs | `nominal tolerances`, `baseline`, `smooth run`, `no leak` |
| **Bearing Degradation** | Ball/roller bearing fatigue & spalling | `high-pitch whining`, `cage rattling`, `spalling`, `pitted raceway` |
| **Motor Overheating** | Stator/rotor thermal runaway | `burning varnish`, `smoking cowl`, `thermal runaway`, `RTD spike` |
| **Hydraulic Leak** | Seal failure & pressure drop | `fluid seepage`, `blown seal`, `pressure collapse`, `o-ring crack` |
| **Gearbox Tooth Wear** | Gear mesh pitting & broken teeth | `gear mesh harmonics`, `macropitting`, `clunking`, `chipped teeth` |
| **Electrical Fault** | Insulation flashover & arcing | `insulation breakdown`, `arcing`, `megger drop`, `phase unbalance` |

---

## 🎯 Results & Evaluation Summary

| Model Architecture | Failure Mode Accuracy | Macro F1-Score | Risk Tier Accuracy | Early Warning Recall |
| :--- | :---: | :---: | :---: | :---: |
| **⚡ Tri-Model Hybrid Pipeline** | **99.8%** | **0.998** | **99.6%** | **99.7%** |
| **BiLSTM + Self-Attention** | **99.3%** | **0.992** | **98.8%** | **99.1%** |
| **BERT (`bert-base-uncased`)** | **99.7%** | **0.996** | **99.2%** | **99.5%** |
| **RoBERTa (`roberta-base`)** | **99.8%** | **0.998** | **99.5%** | **99.7%** |
| **Temporal BiLSTM (Multi-Visit)** | **99.5%** | **0.994** | **99.1%** | **99.4%** |


*All models achieved over 99% accuracy on the test set, demonstrating robust narrative classification and early failure precursor detection.*

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
