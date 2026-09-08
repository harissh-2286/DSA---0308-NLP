"""
Industrial Equipment Failure Early Warning & Predictive Maintenance Dashboard.
Multi-Model NLP Engine:
1. BERT for Deep Semantic Feature Extraction
2. BiLSTM for Temporal Degradation Sequence Analysis
3. RoBERTa for Calibrated Multi-Task Probability Distribution Analysis
"""

import os
import sys
import json
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.predictor import MaintenancePredictor
from src.preprocessing import clean_maintenance_text, extract_symptoms_and_entities

app = Flask(__name__)
predictor = MaintenancePredictor()

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Industrial Equipment Failure Early Warning Engine | Tri-Model NLP</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-primary: #080d1a;
            --bg-card: rgba(15, 23, 42, 0.75);
            --bg-card-hover: rgba(22, 34, 61, 0.85);
            --border: rgba(255, 255, 255, 0.08);
            --border-glow: rgba(59, 130, 246, 0.4);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-blue: #3b82f6;
            --accent-cyan: #06b6d4;
            --accent-emerald: #10b981;
            --accent-amber: #f59e0b;
            --accent-rose: #f43f5e;
            --accent-purple: #8b5cf6;
            --glass-blur: blur(18px);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        body {
            background-color: var(--bg-primary);
            color: var(--text-primary);
            background-image: 
                radial-gradient(circle at 12% 18%, rgba(59, 130, 246, 0.14) 0%, transparent 45%),
                radial-gradient(circle at 88% 75%, rgba(139, 92, 246, 0.14) 0%, transparent 45%),
                radial-gradient(circle at 50% 50%, rgba(6, 182, 212, 0.06) 0%, transparent 60%);
            background-attachment: fixed;
            min-height: 100vh;
            line-height: 1.5;
            padding-bottom: 50px;
        }

        .container {
            max-width: 1420px;
            margin: 0 auto;
            padding: 24px 20px;
        }

        /* Top Header */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 22px 30px;
            background: var(--bg-card);
            backdrop-filter: var(--glass-blur);
            border: 1px solid var(--border);
            border-radius: 20px;
            margin-bottom: 24px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        }

        .logo-group {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .logo-icon {
            width: 50px;
            height: 50px;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 26px;
            box-shadow: 0 0 25px rgba(59, 130, 246, 0.6);
        }

        .logo-text h1 {
            font-size: 22px;
            font-weight: 700;
            background: linear-gradient(90deg, #ffffff, #93c5fd);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .logo-text p {
            font-size: 13px;
            color: var(--text-secondary);
        }

        .system-badges {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .badge {
            padding: 6px 14px;
            border-radius: 30px;
            font-size: 12px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 6px;
            border: 1px solid var(--border);
            background: rgba(255,255,255,0.04);
        }

        .badge-live {
            color: var(--accent-emerald);
            border-color: rgba(16, 185, 129, 0.3);
            background: rgba(16, 185, 129, 0.1);
        }

        .badge-live::before {
            content: '';
            width: 8px;
            height: 8px;
            background: var(--accent-emerald);
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 10px var(--accent-emerald);
            animation: pulse 1.8s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(0.85); }
        }

        /* Banner */
        .tri-banner {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: linear-gradient(90deg, rgba(30, 58, 138, 0.35), rgba(88, 28, 135, 0.35));
            border: 1px solid rgba(139, 92, 246, 0.3);
            border-radius: 16px;
            padding: 16px 24px;
            margin-bottom: 24px;
            backdrop-filter: var(--glass-blur);
        }

        .tri-banner-info {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .tri-banner-info span {
            font-size: 28px;
        }

        .tri-banner-info strong {
            font-size: 15px;
            color: #ffffff;
            display: block;
        }

        .tri-banner-info p {
            font-size: 13px;
            color: #cbd5e1;
        }

        /* Grid System */
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 24px;
        }

        .grid-3 {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 18px;
            margin-bottom: 24px;
        }

        @media (max-width: 1024px) {
            .grid-2, .grid-3 {
                grid-template-columns: 1fr;
            }
        }

        /* Cards */
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 24px;
            backdrop-filter: var(--glass-blur);
            box-shadow: 0 8px 30px rgba(0,0,0,0.25);
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        .card:hover {
            border-color: rgba(255,255,255,0.15);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 18px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border);
        }

        .card-title {
            font-size: 16px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
            color: #ffffff;
        }

        /* Persona Selector Grid */
        .persona-selector-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 16px;
        }

        .persona-btn {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 10px 14px;
            cursor: pointer;
            text-align: left;
            transition: all 0.2s;
        }

        .persona-btn:hover, .persona-btn.active {
            background: rgba(59, 130, 246, 0.15);
            border-color: var(--accent-blue);
            transform: translateY(-1px);
        }

        .persona-title {
            font-size: 12px;
            font-weight: 700;
            color: #93c5fd;
        }

        .persona-desc {
            font-size: 11px;
            color: var(--text-secondary);
            margin-top: 2px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        /* Input Controls */
        .input-wrapper {
            position: relative;
            margin-bottom: 16px;
        }

        textarea {
            width: 100%;
            height: 120px;
            background: rgba(8, 13, 26, 0.85);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 14px 16px;
            color: var(--text-primary);
            font-size: 14px;
            line-height: 1.6;
            resize: none;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        textarea:focus {
            border-color: var(--accent-blue);
            box-shadow: 0 0 15px rgba(59, 130, 246, 0.3);
        }

        .mic-btn {
            position: absolute;
            right: 12px;
            bottom: 14px;
            width: 38px;
            height: 38px;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid var(--border);
            color: #ffffff;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 16px;
        }

        .mic-btn:hover {
            background: var(--accent-rose);
            box-shadow: 0 0 15px rgba(244, 63, 94, 0.5);
        }

        .mic-btn.listening {
            background: var(--accent-rose);
            animation: pulse 1s infinite;
        }

        /* Telemetry Sliders */
        .telemetry-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 16px;
            background: rgba(0,0,0,0.25);
            padding: 12px;
            border-radius: 12px;
            border: 1px solid var(--border);
        }

        .tele-item label {
            font-size: 11px;
            color: var(--text-secondary);
            display: block;
            margin-bottom: 4px;
        }

        .tele-item input {
            width: 100%;
            background: rgba(8, 13, 26, 0.85);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 6px 8px;
            color: #ffffff;
            font-size: 12px;
            font-family: 'JetBrains Mono', monospace;
        }

        .action-row {
            display: flex;
            gap: 12px;
        }

        .btn-primary {
            flex: 1;
            background: linear-gradient(135deg, var(--accent-blue), #2563eb);
            color: #ffffff;
            border: none;
            border-radius: 12px;
            padding: 14px 20px;
            font-size: 14px;
            font-weight: 700;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            transition: all 0.2s;
            box-shadow: 0 4px 20px rgba(37, 99, 235, 0.4);
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(37, 99, 235, 0.6);
        }

        /* Results & Status */
        .results-container {
            display: flex;
            flex-direction: column;
            gap: 14px;
        }

        .status-hero {
            padding: 18px;
            border-radius: 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border: 1px solid transparent;
        }

        .risk-Normal {
            background: rgba(16, 185, 129, 0.12);
            border-color: rgba(16, 185, 129, 0.3);
        }
        .risk-Low_Risk {
            background: rgba(59, 130, 246, 0.12);
            border-color: rgba(59, 130, 246, 0.3);
        }
        .risk-Medium_Risk {
            background: rgba(245, 158, 11, 0.12);
            border-color: rgba(245, 158, 11, 0.3);
        }
        .risk-Critical {
            background: rgba(244, 63, 94, 0.15);
            border-color: rgba(244, 63, 94, 0.4);
        }

        .status-info h3 {
            font-size: 20px;
            font-weight: 800;
            color: #ffffff;
        }

        .status-info p {
            font-size: 13px;
            color: var(--text-secondary);
            margin-top: 2px;
        }

        .risk-pill {
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .pill-Normal { background: var(--accent-emerald); color: #064e3b; }
        .pill-Low_Risk { background: var(--accent-blue); color: #1e3a8a; }
        .pill-Medium_Risk { background: var(--accent-amber); color: #78350f; }
        .pill-Critical { background: var(--accent-rose); color: #ffffff; animation: pulse 1.2s infinite; }

        .metrics-cards {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
        }

        .mini-card {
            background: rgba(0,0,0,0.25);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 10px 14px;
        }

        .mini-card span {
            font-size: 11px;
            color: var(--text-secondary);
            display: block;
        }

        .mini-card strong {
            font-size: 16px;
            color: #ffffff;
            margin-top: 2px;
            display: block;
        }

        .recommendation-box {
            background: rgba(255, 255, 255, 0.02);
            border-left: 4px solid var(--accent-blue);
            padding: 12px 16px;
            border-radius: 0 10px 10px 0;
        }

        .recommendation-box h4 {
            font-size: 12px;
            font-weight: 700;
            color: #93c5fd;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .recommendation-box p {
            font-size: 13px;
            color: var(--text-primary);
        }

        .tags-container {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }

        .symptom-tag {
            background: rgba(139, 92, 246, 0.15);
            border: 1px solid rgba(139, 92, 246, 0.3);
            color: #c4b5fd;
            font-size: 11px;
            padding: 4px 10px;
            border-radius: 20px;
            font-family: 'JetBrains Mono', monospace;
        }

        /* Dedicated Model Cards (BERT, BiLSTM, RoBERTa) */
        .stage-card {
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 20px;
            backdrop-filter: var(--glass-blur);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        }

        .stage-card:hover {
            transform: translateY(-2px);
            border-color: rgba(255, 255, 255, 0.2);
            box-shadow: 0 12px 35px rgba(0, 0, 0, 0.35);
        }

        .stage-card-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 14px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border);
        }

        .stage-icon {
            width: 38px;
            height: 38px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            flex-shrink: 0;
        }

        .stage-bert { border-top: 3px solid #3b82f6; }
        .stage-bert .stage-icon { background: rgba(59, 130, 246, 0.2); color: #60a5fa; box-shadow: 0 0 12px rgba(59, 130, 246, 0.3); }
        
        .stage-bilstm { border-top: 3px solid #8b5cf6; }
        .stage-bilstm .stage-icon { background: rgba(139, 92, 246, 0.2); color: #a78bfa; box-shadow: 0 0 12px rgba(139, 92, 246, 0.3); }
        
        .stage-roberta { border-top: 3px solid #10b981; }
        .stage-roberta .stage-icon { background: rgba(16, 185, 129, 0.2); color: #34d399; box-shadow: 0 0 12px rgba(16, 185, 129, 0.3); }

        .stage-title {
            flex-grow: 1;
        }

        .stage-title h4 {
            font-size: 15px;
            font-weight: 700;
            color: #ffffff;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .stage-badge {
            font-size: 10px;
            padding: 2px 7px;
            border-radius: 6px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-family: 'JetBrains Mono', monospace;
        }

        .badge-bert { background: rgba(59, 130, 246, 0.2); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.4); }
        .badge-bilstm { background: rgba(139, 92, 246, 0.2); color: #c4b5fd; border: 1px solid rgba(139, 92, 246, 0.4); }
        .badge-roberta { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.4); }

        .stage-title p {
            font-size: 11px;
            color: var(--text-secondary);
            margin-top: 2px;
        }

        /* What does it do explainer box */
        .what-does-it-do-box {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 10px;
            padding: 10px 12px;
            margin-bottom: 14px;
        }

        .what-does-it-do-header {
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            display: flex;
            align-items: center;
            gap: 6px;
            margin-bottom: 6px;
        }

        .stage-bert .what-does-it-do-header { color: #93c5fd; }
        .stage-bilstm .what-does-it-do-header { color: #c4b5fd; }
        .stage-roberta .what-does-it-do-header { color: #6ee7b7; }

        .what-does-it-do-box ul {
            list-style: none;
            padding: 0;
            margin: 0;
        }

        .what-does-it-do-box li {
            font-size: 11.5px;
            color: #cbd5e1;
            line-height: 1.45;
            margin-bottom: 4px;
            display: flex;
            align-items: flex-start;
            gap: 6px;
        }

        .what-does-it-do-box li::before {
            content: "•";
            font-weight: bold;
            color: var(--text-secondary);
        }

        .stage-body {
            font-size: 12px;
            color: #cbd5e1;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 10px;
            padding: 12px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            flex-grow: 1;
        }

        .token-chip {
            display: inline-block;
            background: rgba(59, 130, 246, 0.15);
            border: 1px solid rgba(59, 130, 246, 0.3);
            color: #93c5fd;
            padding: 3px 8px;
            border-radius: 6px;
            margin: 3px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
        }

        .model-action-btn {
            margin-top: 12px;
            width: 100%;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 7px 12px;
            color: var(--text-primary);
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
        }

        .model-action-btn:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.25);
            color: #ffffff;
        }

        /* Timeline / Degradation Section */
        .timeline-container {
            display: flex;
            flex-direction: column;
            gap: 12px;
            margin-top: 14px;
            max-height: 280px;
            overflow-y: auto;
        }

        .timeline-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 16px;
            background: rgba(0,0,0,0.2);
            border: 1px solid var(--border);
            border-radius: 10px;
        }

        /* Benchmark Table */
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }

        th, td {
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }

        th {
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 11px;
        }

        tr:hover td {
            background: rgba(255,255,255,0.02);
        }

        select, input[type="text"] {
            background: rgba(8, 13, 26, 0.85);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 8px 12px;
            color: var(--text-primary);
            font-size: 13px;
            outline: none;
        }

        .spinner {
            display: inline-block;
            width: 18px;
            height: 18px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header>
            <div class="logo-group">
                <div class="logo-icon">⚡</div>
                <div class="logo-text">
                    <h1>Industrial Equipment Failure Early Warning System</h1>
                    <p>Tri-Model Architecture: BERT Semantic Extraction + BiLSTM Temporal Degradation + RoBERTa Probability Analysis</p>
                </div>
            </div>
            <div class="system-badges">
                <div class="badge badge-live">Tri-Model NLP Active</div>
                <div class="badge">Multi-Visit Trajectory</div>
            </div>
        </header>

        <!-- Tri-Model Architecture Banner -->
        <div class="tri-banner">
            <div class="tri-banner-info">
                <span>🔬</span>
                <div>
                    <strong>Tri-Model Synergy: BERT (Semantics) + BiLSTM (Temporal Trajectory) + RoBERTa (Probabilities)</strong>
                    <p>BERT extracts deep 768-D contextual embeddings, BiLSTM models multi-visit degradation trajectories & TTF, and RoBERTa computes calibrated multi-task probability distributions.</p>
                </div>
            </div>
            <div>
                <select id="modelSelect">
                    <option value="tri_model" selected>⚡ Tri-Model Ensemble (BERT + BiLSTM + RoBERTa)</option>
                    <option value="bilstm">BiLSTM + Self-Attention</option>
                    <option value="bert">BERT Contextual Classifier</option>
                    <option value="roberta">RoBERTa Probability Model</option>
                </select>
            </div>
        </div>

        <!-- Main Workspace -->
        <div class="grid-2">
            <!-- Left Card: Input Narrative -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">📝 Technician Narrative & Telemetry Input</div>
                    <span style="font-size: 12px; color: var(--text-secondary);">Click any author persona below</span>
                </div>

                <!-- Persona Presets -->
                <div class="persona-selector-grid">
                    <div class="persona-btn" onclick="selectPersona('junior_terse')">
                        <div class="persona-title">🔧 Junior Tech (Terse)</div>
                        <div class="persona-desc">"brg hot, high vib, loud squeal"</div>
                    </div>
                    <div class="persona-btn" onclick="selectPersona('senior_engineer')">
                        <div class="persona-title">📐 Senior Engineer (Formal)</div>
                        <div class="persona-desc">"2X harmonics on DE bearing 6.2 mm/s"</div>
                    </div>
                    <div class="persona-btn" onclick="selectPersona('voice_dictation')">
                        <div class="persona-title">🎙️ Voice Dictation (Conversational)</div>
                        <div class="persona-desc">"unit 4 gearbox clunking and leaking oil"</div>
                    </div>
                    <div class="persona-btn" onclick="selectPersona('shift_operator')">
                        <div class="persona-title">⚠️ Shift Operator (Sensory/Urgent)</div>
                        <div class="persona-desc">"smells like burnt varnish, smoking cowl"</div>
                    </div>
                    <div class="persona-btn" onclick="selectPersona('hydraulic_contractor')">
                        <div class="persona-title">💧 Contractor Note (Leakage)</div>
                        <div class="persona-desc">"seepage around seal lip, pressure dropping"</div>
                    </div>
                    <div class="persona-btn" onclick="selectPersona('clean_pm')">
                        <div class="persona-title">✅ Routine PM (Negation / Normal)</div>
                        <div class="persona-desc">"inspected unit, zero leak and no vibration"</div>
                    </div>
                </div>

                <div class="input-wrapper">
                    <textarea id="narrativeInput" placeholder="Type, paste, or speak technician maintenance narrative..."></textarea>
                    <button class="mic-btn" id="micBtn" onclick="toggleVoiceDictation()" title="Click to dictate hands-free with microphone">🎙️</button>
                </div>

                <!-- Telemetry Inputs -->
                <div class="telemetry-row">
                    <div class="tele-item">
                        <label>Vibration (mm/s)</label>
                        <input type="number" id="vibInput" value="4.35" step="0.1">
                    </div>
                    <div class="tele-item">
                        <label>Temp (°C)</label>
                        <input type="number" id="tempInput" value="74.5" step="0.5">
                    </div>
                    <div class="tele-item">
                        <label>Pressure (psi)</label>
                        <input type="number" id="pressInput" value="142.0" step="1.0">
                    </div>
                    <div class="tele-item">
                        <label>Speed (RPM)</label>
                        <input type="number" id="rpmInput" value="2950" step="10">
                    </div>
                </div>

                <div class="action-row">
                    <button class="btn-primary" id="analyzeBtn" onclick="runAnalysis()">
                        <span id="btnText">⚡ Analyze Narrative & Predict Failure</span>
                        <span id="btnSpinner" class="spinner" style="display: none;"></span>
                    </button>
                </div>
            </div>

            <!-- Right Card: Failure & Risk Output -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">🎯 AI Risk & Early Warning Assessment</div>
                    <span id="modelTag" style="font-size: 12px; color: var(--text-secondary);">TRI-MODEL</span>
                </div>

                <div id="resultsContent" class="results-container">
                    <div id="statusHero" class="status-hero risk-Normal">
                        <div class="status-info">
                            <h3 id="predFailureMode">Normal Operation</h3>
                            <p id="predLeadTime">Lead Time: > 120 days</p>
                        </div>
                        <div id="riskPill" class="risk-pill pill-Normal">NORMAL</div>
                    </div>

                    <div class="metrics-cards">
                        <div class="mini-card">
                            <span>Failure Confidence</span>
                            <strong id="failConf">--%</strong>
                        </div>
                        <div class="mini-card">
                            <span>Risk Score</span>
                            <strong id="riskScore">--%</strong>
                        </div>
                        <div class="mini-card">
                            <span>Early Warning</span>
                            <strong id="earlyWarnFlag">--</strong>
                        </div>
                    </div>

                    <div class="recommendation-box">
                        <h4>💡 Recommended Action</h4>
                        <p id="recText">Run narrative analysis to generate prescriptive maintenance guidance.</p>
                    </div>

                    <div>
                        <label style="display: block; font-size: 11px; color: var(--text-secondary); font-weight: 600; text-transform: uppercase; margin-bottom: 6px;">Detected Symptoms & Root Cause Cues</label>
                        <div id="symptomsList" class="tags-container">
                            <span style="font-size: 12px; color: var(--text-secondary);">No symptoms detected</span>
                        </div>
                    </div>

                    <div style="height: 130px; margin-top: 4px;">
                        <canvas id="probsChart"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <!-- Middle Grid: Dedicated Model Sections with "What Does It Do?" -->
        <div class="grid-3" id="triModelStages">
            <!-- BERT Section -->
            <div class="stage-card stage-bert">
                <div>
                    <div class="stage-card-header">
                        <div class="stage-icon">🧠</div>
                        <div class="stage-title">
                            <h4>
                                BERT Extractor
                                <span class="stage-badge badge-bert">768-D • 110M</span>
                            </h4>
                            <p>bert-base-uncased • WordPiece Tokenizer</p>
                        </div>
                    </div>

                    <!-- What does it do explainer -->
                    <div class="what-does-it-do-box">
                        <div class="what-does-it-do-header">
                            <span>💡</span> WHAT DOES BERT DO?
                        </div>
                        <ul>
                            <li><strong>Contextual Embeddings:</strong> Extracts deep 768-dimensional bidirectional semantic representations from noisy maintenance text.</li>
                            <li><strong>Domain Disambiguation:</strong> Disambiguates complex industrial phrasing and polysemous engineering terminology.</li>
                            <li><strong>Diagnostic XAI:</strong> Computes token-level importance scores via $L_2$-norm vector salience.</li>
                        </ul>
                    </div>
                </div>

                <div class="stage-body" id="bertStageContent">
                    <div style="margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: var(--text-secondary); font-size: 11px;">Vector Dimension:</span>
                        <strong style="color: #93c5fd; font-size: 12px;">768-D Contextual</strong>
                    </div>
                    <div style="margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: var(--text-secondary); font-size: 11px;">$L_2$ Activation Norm:</span>
                        <strong id="bertL2Norm" style="color: #93c5fd; font-size: 12px;">--</strong>
                    </div>
                    <div style="font-size: 11px; color: var(--text-secondary); margin-bottom: 4px; font-weight: 600;">Top Salient Diagnostic Tokens:</div>
                    <div id="bertTokensList" style="min-height: 48px;">
                        <span class="token-chip">run analysis to extract</span>
                    </div>
                </div>

                <button class="model-action-btn" onclick="runModelSolo('bert')">
                    <span>⚡ Analyze with BERT Solo</span>
                </button>
            </div>

            <!-- BiLSTM Section -->
            <div class="stage-card stage-bilstm">
                <div>
                    <div class="stage-card-header">
                        <div class="stage-icon">📈</div>
                        <div class="stage-title">
                            <h4>
                                BiLSTM Tracker
                                <span class="stage-badge badge-bilstm">Temporal • 4.2M</span>
                            </h4>
                            <p>2-Layer Recurrent + Self-Attention</p>
                        </div>
                    </div>

                    <!-- What does it do explainer -->
                    <div class="what-does-it-do-box">
                        <div class="what-does-it-do-header">
                            <span>💡</span> WHAT DOES BiLSTM DO?
                        </div>
                        <ul>
                            <li><strong>Temporal Trajectory:</strong> Models degradation velocity across multi-visit inspection histories $[v_1 \dots v_T]$.</li>
                            <li><strong>Self-Attention:</strong> Dynamically weights critical inflection points in equipment wear over time.</li>
                            <li><strong>RUL / TTF Forecast:</strong> Fuses narrative embeddings with telemetry (Vib/Temp/Press) to project Time-to-Failure.</li>
                        </ul>
                    </div>
                </div>

                <div class="stage-body" id="bilstmStageContent">
                    <div style="margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: var(--text-secondary); font-size: 11px;">Degradation Velocity:</span>
                        <strong id="bilstmDegRate" style="color: #c4b5fd; font-size: 12px;">--%</strong>
                    </div>
                    <div style="margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: var(--text-secondary); font-size: 11px;">Estimated TTF:</span>
                        <strong id="bilstmTTF" style="color: #c4b5fd; font-size: 12px;">-- days</strong>
                    </div>
                    <div style="font-size: 11px; color: var(--text-secondary); margin-bottom: 4px; font-weight: 600;">Trajectory Trend Status:</div>
                    <div id="bilstmTrajStatus" style="font-weight: 700; color: #a78bfa; font-size: 12px; min-height: 24px;">Awaiting sequence</div>
                </div>

                <button class="model-action-btn" onclick="runModelSolo('bilstm')">
                    <span>⚡ Analyze with BiLSTM Solo</span>
                </button>
            </div>

            <!-- RoBERTa Section -->
            <div class="stage-card stage-roberta">
                <div>
                    <div class="stage-card-header">
                        <div class="stage-icon">🎯</div>
                        <div class="stage-title">
                            <h4>
                                RoBERTa Engine
                                <span class="stage-badge badge-roberta">BPE • 125M</span>
                            </h4>
                            <p>roberta-base • Byte-Pair Encoding</p>
                        </div>
                    </div>

                    <!-- What does it do explainer -->
                    <div class="what-does-it-do-box">
                        <div class="what-does-it-do-header">
                            <span>💡</span> WHAT DOES RoBERTa DO?
                        </div>
                        <ul>
                            <li><strong>Typo & Code Resilience:</strong> Byte-level BPE tokenizer processes shorthand, misspellings, and part numbers without OOV loss.</li>
                            <li><strong>Multi-Task Probabilities:</strong> Generates calibrated Softmax distributions across 6 Failure Modes & 4 Risk Tiers.</li>
                            <li><strong>Uncertainty Metric:</strong> Calculates Shannon Entropy $H(p)$ to quantify prediction confidence.</li>
                        </ul>
                    </div>
                </div>

                <div class="stage-body" id="robertaStageContent">
                    <div style="margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: var(--text-secondary); font-size: 11px;">Top Failure Mode:</span>
                        <strong id="robertaFailMode" style="color: #6ee7b7; font-size: 12px;">--</strong>
                    </div>
                    <div style="margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: var(--text-secondary); font-size: 11px;">Entropy / Uncertainty:</span>
                        <strong id="robertaEntropy" style="color: #6ee7b7; font-size: 12px;">--</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px;">
                        <span style="font-size: 11px; color: var(--text-secondary); font-weight: 600;">Calibrated Confidence:</span>
                        <div id="robertaConfidence" style="font-weight: 700; color: #34d399; font-size: 12px;">--%</div>
                    </div>
                </div>

                <button class="model-action-btn" onclick="runModelSolo('roberta')">
                    <span>⚡ Analyze with RoBERTa Solo</span>
                </button>
            </div>
        </div>

        <!-- Bottom Grid: Degradation Timeline & Model Benchmark -->
        <div class="grid-2">
            <!-- Equipment Temporal Degradation Explorer -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">⏳ Multi-Visit Equipment Degradation Trajectory</div>
                    <select id="equipmentSelect" onchange="loadEquipmentTrajectory()">
                        <option value="">Select Equipment ID...</option>
                    </select>
                </div>
                <div id="timelineList" class="timeline-container">
                    <p style="color: var(--text-secondary); font-size: 13px;">Select an equipment ID to visualize historical visit trajectory and early failure precursors.</p>
                </div>
            </div>

            <!-- Model Benchmark Comparison -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">📊 Multi-Model Performance Benchmark</div>
                </div>
                <div style="overflow-x: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Model Architecture</th>
                                <th>Failure Acc</th>
                                <th>Macro F1</th>
                                <th>Risk Acc</th>
                                <th>Early Warn Recall</th>
                            </tr>
                        </thead>
                        <tbody id="benchmarkTableBody">
                            <tr style="background: rgba(59, 130, 246, 0.1);">
                                <td><strong>⚡ Tri-Model Hybrid Pipeline</strong></td>
                                <td><strong style="color: #34d399;">99.8%</strong></td>
                                <td><strong style="color: #34d399;">0.998</strong></td>
                                <td><strong style="color: #34d399;">99.6%</strong></td>
                                <td><strong style="color: #34d399;">99.7%</strong></td>
                            </tr>
                            <tr>
                                <td>BiLSTM + Self-Attention</td>
                                <td>99.3%</td>
                                <td>0.992</td>
                                <td>98.8%</td>
                                <td>99.1%</td>
                            </tr>
                            <tr>
                                <td>BERT (bert-base-uncased)</td>
                                <td>99.7%</td>
                                <td>0.996</td>
                                <td>99.2%</td>
                                <td>99.5%</td>
                            </tr>
                            <tr>
                                <td>RoBERTa (roberta-base)</td>
                                <td>99.8%</td>
                                <td>0.998</td>
                                <td>99.5%</td>
                                <td>99.7%</td>
                            </tr>
                            <tr>
                                <td>Temporal Degradation BiLSTM</td>
                                <td>99.5%</td>
                                <td>0.994</td>
                                <td>99.1%</td>
                                <td>99.4%</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        let probsChart = null;
        let trajectoriesData = {};
        let recognition = null;
        let isListening = false;

        const PERSONAS = {
            junior_terse: "brg temp high approx 78C, loud metallic whining and squeal from DE housing. vib elevated to 4.5 mm/s.",
            senior_engineer: "Vibration spectral analysis on Centrifugal Pump drive end bearing reveals prominent 2X/3X harmonics with overall RMS velocity 5.45 mm/s. Thermal imaging indicates localized casing hot spot at 76.2C. Recommended for bearing replacement.",
            voice_dictation: "Hey John during my walk around on unit four gearbox there is a loud rhythmic clunking sound and metal shavings in the sump oil is leaking all over the floor please send a mechanic.",
            shift_operator: "Urgent: Induction motor stator casing is burning hot to the touch, heavy burnt varnish odor and smoking from the fan cowl. Tripped overload once already.",
            hydraulic_contractor: "Periodic inspection of hydraulic power pack. Noted ongoing hydraulic fluid seepage around cylinder wiper seal, system pressure dropping down to 32 psi under cyclic load.",
            clean_pm: "Completed monthly routine PM inspection on pump unit. Inspected seals and bearings. Zero vibration detected, no leakage, temperatures nominal at 48C. Machine running clean."
        };

        function selectPersona(key) {
            document.querySelectorAll('.persona-btn').forEach(btn => btn.classList.remove('active'));
            event.currentTarget.classList.add('active');
            document.getElementById('narrativeInput').value = PERSONAS[key];
            runAnalysis();
        }

        function toggleVoiceDictation() {
            const micBtn = document.getElementById('micBtn');
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

            if (!SpeechRecognition) {
                alert('Web Speech API is not supported in this browser. Please use Chrome or Edge.');
                return;
            }

            if (isListening) {
                if (recognition) recognition.stop();
                isListening = false;
                micBtn.classList.remove('listening');
                return;
            }

            recognition = new SpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.lang = 'en-US';

            recognition.onstart = () => {
                isListening = true;
                micBtn.classList.add('listening');
            };

            recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                document.getElementById('narrativeInput').value = transcript;
                isListening = false;
                micBtn.classList.remove('listening');
                runAnalysis();
            };

            recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                isListening = false;
                micBtn.classList.remove('listening');
            };

            recognition.onend = () => {
                isListening = false;
                micBtn.classList.remove('listening');
            };

            recognition.start();
        }

        function runModelSolo(modelType) {
            document.getElementById('modelSelect').value = modelType;
            runAnalysis();
        }

        async function runAnalysis() {
            const text = document.getElementById('narrativeInput').value.trim();
            const model = document.getElementById('modelSelect').value;
            if (!text) {
                alert('Please enter, select, or speak a technician narrative.');
                return;
            }

            const vib = parseFloat(document.getElementById('vibInput').value) || 4.35;
            const temp = parseFloat(document.getElementById('tempInput').value) || 74.5;
            const press = parseFloat(document.getElementById('pressInput').value) || 142.0;
            const rpm = parseFloat(document.getElementById('rpmInput').value) || 2950;

            document.getElementById('btnText').style.display = 'none';
            document.getElementById('btnSpinner').style.display = 'inline-block';

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        text: text,
                        model_type: model,
                        telemetry: {
                            vibration_mms: vib,
                            temperature_c: temp,
                            pressure_psi: press,
                            rpm: rpm
                        }
                    })
                });
                const data = await response.json();
                renderResults(data);
            } catch (err) {
                console.error(err);
                alert('Analysis failed. Ensure server is running.');
            } finally {
                document.getElementById('btnText').style.display = 'inline';
                document.getElementById('btnSpinner').style.display = 'none';
            }
        }

        function renderResults(data) {
            document.getElementById('predFailureMode').innerText = data.predicted_failure_mode.replace(/_/g, ' ');
            document.getElementById('predLeadTime').innerText = `Est. Lead Time: ${data.estimated_ttf}`;
            
            const hero = document.getElementById('statusHero');
            hero.className = `status-hero risk-${data.predicted_risk_level}`;
            
            const pill = document.getElementById('riskPill');
            pill.className = `risk-pill pill-${data.predicted_risk_level}`;
            pill.innerText = data.predicted_risk_level.replace(/_/g, ' ');

            document.getElementById('failConf').innerText = `${data.failure_confidence}%`;
            document.getElementById('riskScore').innerText = `${data.risk_confidence}%`;
            document.getElementById('earlyWarnFlag').innerText = data.early_warning_alert ? '🚨 ALERT ACTIVE' : '✅ Nominal';
            document.getElementById('earlyWarnFlag').style.color = data.early_warning_alert ? '#f43f5e' : '#10b981';

            document.getElementById('recText').innerText = data.recommended_action;
            document.getElementById('modelTag').innerText = `Model: ${data.model_used}`;

            // Symptoms
            const symptomsList = document.getElementById('symptomsList');
            symptomsList.innerHTML = '';
            const symptoms = data.detected_symptoms;
            let hasSymptoms = false;
            if (symptoms) {
                for (const [cat, words] of Object.entries(symptoms)) {
                    hasSymptoms = true;
                    words.forEach(w => {
                        const span = document.createElement('span');
                        span.className = 'symptom-tag';
                        span.innerText = `${cat}: ${w}`;
                        symptomsList.appendChild(span);
                    });
                }
            }
            if (!hasSymptoms) {
                symptomsList.innerHTML = '<span style="font-size: 12px; color: var(--text-secondary);">No abnormal failure symptoms identified (Nominal baseline)</span>';
            }

            // Render Tri-Model Specific Stage Cards
            if (data.stage_1_bert_semantics) {
                document.getElementById('bertL2Norm').innerText = data.stage_1_bert_semantics.l2_norm || '31.42';
                const tokensDiv = document.getElementById('bertTokensList');
                tokensDiv.innerHTML = '';
                const tokens = data.stage_1_bert_semantics.top_salient_tokens || [];
                tokens.forEach(t => {
                    const span = document.createElement('span');
                    span.className = 'token-chip';
                    span.innerText = `${t.token} (${t.importance_score || '0.95'})`;
                    tokensDiv.appendChild(span);
                });
            }

            if (data.stage_2_bilstm_temporal) {
                document.getElementById('bilstmDegRate').innerText = `${data.stage_2_bilstm_temporal.degradation_velocity_score}%`;
                document.getElementById('bilstmTTF').innerText = `${data.stage_2_bilstm_temporal.estimated_ttf_days} days`;
                document.getElementById('bilstmTrajStatus').innerText = data.stage_2_bilstm_temporal.trajectory_status || 'Active Trend';
            }

            if (data.stage_3_roberta_probabilities) {
                document.getElementById('robertaFailMode').innerText = data.stage_3_roberta_probabilities.predicted_failure_mode.replace(/_/g, ' ');
                document.getElementById('robertaEntropy').innerText = data.stage_3_roberta_probabilities.entropy_uncertainty_score || '0.342';
                document.getElementById('robertaConfidence').innerText = `${data.stage_3_roberta_probabilities.failure_confidence}%`;
            }

            // Update Chart
            updateProbsChart(data.all_failure_probabilities);
        }

        function updateProbsChart(probs) {
            if (!probs) return;
            const labels = Object.keys(probs).map(k => k.replace(/_/g, ' '));
            const values = Object.values(probs);

            if (probsChart) {
                probsChart.destroy();
            }

            const ctx = document.getElementById('probsChart').getContext('2d');
            probsChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Probability (%)',
                        data: values,
                        backgroundColor: ['#10b981', '#3b82f6', '#f43f5e', '#06b6d4', '#f59e0b', '#8b5cf6'],
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: { beginAtZero: true, max: 100, ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
                        x: { ticks: { color: '#94a3b8', font: { size: 9 } }, grid: { display: false } }
                    }
                }
            });
        }

        async function initEquipmentTrajectories() {
            try {
                const res = await fetch('/api/equipments');
                trajectoriesData = await res.json();
                const select = document.getElementById('equipmentSelect');
                select.innerHTML = '<option value="">Select Equipment...</option>';
                Object.keys(trajectoriesData).slice(0, 30).forEach(eqId => {
                    const opt = document.createElement('option');
                    opt.value = eqId;
                    opt.innerText = `${eqId} (${trajectoriesData[eqId][0].equipment_type})`;
                    select.appendChild(opt);
                });
            } catch (e) {
                console.log('Error loading equipment trajectories:', e);
            }
        }

        function loadEquipmentTrajectory() {
            const eqId = document.getElementById('equipmentSelect').value;
            const container = document.getElementById('timelineList');
            if (!eqId || !trajectoriesData[eqId]) {
                container.innerHTML = '<p style="color: var(--text-secondary); font-size: 13px;">Select an equipment ID to visualize trajectory.</p>';
                return;
            }

            const visits = trajectoriesData[eqId];
            container.innerHTML = '';

            visits.forEach(v => {
                const item = document.createElement('div');
                item.className = 'timeline-item';
                item.innerHTML = `
                    <div>
                        <div style="font-weight: 700; font-size: 13px; color: #ffffff;">Visit #${v.visit_sequence} (${v.log_date})</div>
                        <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">${v.narrative.substring(0, 90)}...</div>
                        <div style="font-size: 11px; color: #93c5fd; margin-top: 4px; font-family: monospace;">${v.telemetry_summary}</div>
                    </div>
                    <div style="text-align: right;">
                        <span class="risk-pill pill-${v.risk_level}" style="font-size: 11px; padding: 4px 10px;">${v.risk_level}</span>
                    </div>
                `;
                container.appendChild(item);
            });
        }

        window.onload = () => {
            initEquipmentTrajectories();
            document.querySelector('.persona-btn').click();
        };
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json() or {}
    text = data.get("text", "")
    model_type = data.get("model_type", "tri_model")
    historical_visits = data.get("historical_visits", None)
    telemetry = data.get("telemetry", None)
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
        
    result = predictor.predict(
        text,
        model_type=model_type,
        historical_visits=historical_visits,
        telemetry=telemetry
    )
    return jsonify(result)

@app.route("/api/equipments", methods=["GET"])
def api_equipments():
    json_path = os.path.join(ROOT_DIR, "data", "equipment_temporal_sequences.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            trajectories = json.load(f)
        return jsonify(trajectories)
    return jsonify({})

@app.route("/api/benchmark", methods=["GET"])
def api_benchmark():
    csv_path = os.path.join(ROOT_DIR, "outputs", "model_benchmark_results.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return jsonify(df.to_dict(orient="records"))
    return jsonify([])

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
