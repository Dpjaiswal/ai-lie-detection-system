"""
AI-Powered Lie Detection System — Streamlit Dashboard
Full-featured UI with text analysis, audio upload, microphone recording,
prediction display, confidence meter, explainability panel, and analytics.
"""
from __future__ import annotations

import io
import os
import time
import json
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ──────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Lie Detection System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-org/lie-detection-system",
        "Report a Bug": "https://github.com/your-org/lie-detection-system/issues",
        "About": "AI-Powered Multimodal Lie Detection System v1.0.0",
    },
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# ──────────────────────────────────────────────────────────────
# Custom CSS — Dark Premium Theme
# ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Imports ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #E0E0E0;
    }
    .stApp {
        background: linear-gradient(135deg, #0A0A14 0%, #121225 50%, #0F172A 100%);
        min-height: 100vh;
    }
    .main .block-container { max-width: 1200px; padding: 2rem; }

    /* ── Custom Scrollbar ── */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.01);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(108,99,255,0.3);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(108,99,255,0.5);
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D0D1F 0%, #06060F 100%);
        border-right: 1px solid rgba(108,99,255,0.15);
    }

    /* ── Cards with Glassmorphism ── */
    .card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(108,99,255,0.18);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0,0,0,0.37);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .card:hover { 
        border-color: rgba(108,99,255,0.5);
        box-shadow: 0 12px 40px 0 rgba(108,99,255,0.15);
        transform: translateY(-2px);
    }

    /* ── Fade-in animation for results ── */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in {
        animation: fadeIn 0.4s ease-out forwards;
    }

    /* ── Verdict badges ── */
    .verdict-lie {
        display: inline-block;
        background: linear-gradient(135deg, #FF4757, #FF6B81);
        color: white; border-radius: 100px;
        padding: 10px 28px; font-size: 1.15rem;
        font-weight: 700; letter-spacing: 0.5px;
        box-shadow: 0 0 25px rgba(255,71,87,0.4);
        animation: pulse 2s infinite;
    }
    .verdict-truth {
        display: inline-block;
        background: linear-gradient(135deg, #2ECC71, #27AE60);
        color: white; border-radius: 100px;
        padding: 10px 28px; font-size: 1.15rem;
        font-weight: 700; letter-spacing: 0.5px;
        box-shadow: 0 0 25px rgba(46,204,113,0.4);
    }
    .verdict-uncertain {
        display: inline-block;
        background: linear-gradient(135deg, #F39C12, #E67E22);
        color: white; border-radius: 100px;
        padding: 10px 28px; font-size: 1.15rem;
        font-weight: 700;
        box-shadow: 0 0 25px rgba(243,156,18,0.4);
    }

    /* ── Pulse animation ── */
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 20px rgba(255,71,87,0.4); }
        50% { box-shadow: 0 0 35px rgba(255,71,87,0.7); }
    }

    /* ── Metric tiles ── */
    .metric-tile {
        background: rgba(108,99,255,0.08);
        border: 1px solid rgba(108,99,255,0.25);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        transition: all 0.2s ease;
    }
    .metric-tile:hover {
        background: rgba(108,99,255,0.12);
        border-color: rgba(108,99,255,0.4);
    }
    .metric-value { font-size: 2.2rem; font-weight: 800; color: #8B7CF6; }
    .metric-label { font-size: 0.85rem; color: #AAA; text-transform: uppercase; letter-spacing: 0.8px; }

    /* ── Disclaimer banner ── */
    .disclaimer {
        background: rgba(255,165,0,0.08);
        border: 1px solid rgba(255,165,0,0.3);
        border-radius: 12px;
        padding: 14px 20px;
        color: #FFA500;
        font-size: 0.85rem;
        line-height: 1.6;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.45rem; font-weight: 700;
        background: linear-gradient(135deg, #8B7CF6, #FF6584);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 14px;
        letter-spacing: -0.3px;
    }

    /* ── Inputs ── */
    .stTextArea textarea {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(108,99,255,0.25) !important;
        border-radius: 12px !important;
        color: #E0E0E0 !important;
        font-size: 0.95rem !important;
        transition: border-color 0.25s ease !important;
    }
    .stTextArea textarea:focus {
        border-color: rgba(108,99,255,0.6) !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6C63FF, #8B7CF6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.6rem 2.2rem !important;
        transition: all 0.3s cubic-bezier(0.2, 0.8, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(108,99,255,0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(108,99,255,0.5) !important;
    }
    .stButton > button:active {
        transform: translateY(0px) !important;
    }
    div[data-testid="stHorizontalBlock"] { gap: 16px; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────

def make_api_call(endpoint: str, **kwargs) -> Optional[Dict]:
    """Make an API call, return None on failure."""
    try:
        response = requests.post(f"{API_BASE_URL}/{endpoint.lstrip('/')}", **kwargs, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API server. Is it running on localhost:8000?")
        return None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None


def _mock_text_response(text: str) -> Dict:
    """Generate mock response when API is not available."""
    np.random.seed(hash(text) % (2**32))
    lie_prob = float(np.random.beta(2, 2))
    truth_prob = 1.0 - lie_prob
    prediction = "lie" if lie_prob > 0.5 else "truth"
    confidence = max(lie_prob, truth_prob)
    verdict = (
        "likely_lie" if lie_prob >= 0.75
        else "possibly_lie" if lie_prob >= 0.55
        else "uncertain" if lie_prob >= 0.45
        else "likely_truth"
    )
    return {
        "text_prediction": prediction,
        "lie_probability": lie_prob,
        "truth_probability": truth_prob,
        "confidence": confidence,
        "model_used": "roberta (demo)",
        "verdict": verdict,
        "processing_time_ms": np.random.uniform(80, 250),
        "explanation": {
            "important_words": [
                {"word": "definitely", "weight": 0.15, "direction": "lie"},
                {"word": "never", "weight": 0.09, "direction": "lie"},
                {"word": "honest", "weight": -0.08, "direction": "truth"},
            ],
            "audio_features": [],
        },
    }


def _mock_audio_response(duration: float = 5.0) -> Dict:
    np.random.seed(int(time.time()) % (2**32))
    stress_prob = float(np.random.beta(2, 2))
    return {
        "voice_prediction": "stress_detected" if stress_prob > 0.5 else "no_stress",
        "stress_probability": stress_prob,
        "confidence": max(stress_prob, 1 - stress_prob),
        "model_used": "random_forest (demo)",
        "audio_duration_seconds": duration,
        "processing_time_ms": np.random.uniform(150, 400),
    }


def _mock_multimodal_response(text: str) -> Dict:
    text_r = _mock_text_response(text)
    audio_r = _mock_audio_response()
    lie_prob = 0.6 * text_r["lie_probability"] + 0.4 * audio_r["stress_probability"]
    verdict = (
        "likely_lie" if lie_prob >= 0.75
        else "possibly_lie" if lie_prob >= 0.55
        else "uncertain" if lie_prob >= 0.45
        else "likely_truth"
    )
    return {
        "text_prediction": text_r["text_prediction"],
        "voice_prediction": audio_r["voice_prediction"],
        "final_prediction": verdict,
        "confidence": max(lie_prob, 1 - lie_prob),
        "lie_probability": lie_prob,
        "text_lie_probability": text_r["lie_probability"],
        "audio_stress_probability": audio_r["stress_probability"],
        "fusion_method": "hybrid",
        "processing_time_ms": text_r["processing_time_ms"] + audio_r["processing_time_ms"],
        "explanation": {
            "important_words": [
                {"word": "definitely", "weight": 0.15, "direction": "lie"},
                {"word": "never", "weight": 0.09, "direction": "lie"},
            ],
            "audio_features": [
                {"feature": "pitch_mean", "shap_value": 0.18, "direction": "stress_detected"},
                {"feature": "jitter_local", "shap_value": 0.14, "direction": "stress_detected"},
            ],
        },
    }


def render_confidence_gauge(confidence: float, lie_prob: float, title: str = "Confidence Meter"):
    """Render a Plotly gauge chart for confidence."""
    color = "#FF4757" if lie_prob > 0.6 else "#FFA500" if lie_prob > 0.45 else "#2ECC71"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(lie_prob * 100, 1),
        title={"text": title, "font": {"size": 16, "color": "#E0E0E0"}},
        number={"suffix": "%", "font": {"color": color, "size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#E0E0E0", "tickfont": {"color": "#888"}},
            "bar": {"color": color},
            "bgcolor": "rgba(255,255,255,0.05)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 45], "color": "rgba(46,204,113,0.15)"},
                {"range": [45, 55], "color": "rgba(255,165,0,0.15)"},
                {"range": [55, 100], "color": "rgba(255,71,87,0.15)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.8,
                "value": 50,
            },
        },
    ))
    fig.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E0E0E0", "family": "Inter"},
        margin={"t": 40, "b": 10, "l": 10, "r": 10},
    )
    return fig


def render_word_importance_chart(word_importances: list):
    """Render horizontal bar chart of word importances."""
    if not word_importances:
        return None
    words = [w["word"] for w in word_importances[:10]]
    weights = [w["weight"] for w in word_importances[:10]]
    colors = ["#FF4757" if w > 0 else "#2ECC71" for w in weights]
    fig = go.Figure(go.Bar(
        x=weights,
        y=words,
        orientation="h",
        marker_color=colors,
        text=[f"{w:+.3f}" for w in weights],
        textposition="outside",
        textfont={"color": "#E0E0E0"},
    ))
    fig.update_layout(
        title="Word-Level Importance (LIME)",
        title_font={"color": "#E0E0E0", "size": 14},
        xaxis={"title": "Weight (+ → lie, - → truth)", "color": "#888"},
        yaxis={"color": "#E0E0E0"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font={"family": "Inter", "color": "#E0E0E0"},
        height=350,
        margin={"l": 80, "r": 60, "t": 40, "b": 40},
        xaxis_zeroline=True,
        xaxis_zerolinecolor="#444",
    )
    return fig


def render_audio_feature_chart(audio_features: list):
    """Render audio feature importance chart."""
    if not audio_features:
        return None
    feats = [f["feature"] for f in audio_features[:8]]
    vals = [f["shap_value"] for f in audio_features[:8]]
    colors = ["#FF4757" if v > 0 else "#2ECC71" for v in vals]
    fig = go.Figure(go.Bar(
        x=feats, y=vals,
        marker_color=colors,
        text=[f"{v:+.3f}" for v in vals],
        textposition="outside",
        textfont={"color": "#E0E0E0"},
    ))
    fig.update_layout(
        title="Audio Feature Importance (SHAP)",
        title_font={"color": "#E0E0E0", "size": 14},
        yaxis={"title": "SHAP Value", "color": "#888"},
        xaxis={"color": "#E0E0E0"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font={"family": "Inter", "color": "#E0E0E0"},
        height=300,
        margin={"l": 40, "r": 40, "t": 40, "b": 80},
    )
    return fig


def render_verdict_badge(verdict: str) -> str:
    if "lie" in verdict.lower():
        label = "DECEPTION LIKELY" if "likely" in verdict else "POSSIBLY DECEPTIVE"
        return f'<div class="verdict-lie">{label}</div>'
    elif "truth" in verdict.lower():
        label = "LIKELY TRUTHFUL" if "likely" in verdict else "POSSIBLY TRUTHFUL"
        return f'<div class="verdict-truth">{label}</div>'
    else:
        return '<div class="verdict-uncertain">UNCERTAIN</div>'


FEATURE_DESCRIPTIONS = {
    "pitch_mean": "Fundamental frequency (F0) average. Higher or fluctuating pitch often correlates with stress or emotional arousal.",
    "jitter_local": "Cycle-to-cycle frequency variation. Indicates micro-tremors in vocal cords, a strong sign of vocal tension.",
    "shimmer_local": "Cycle-to-cycle amplitude variation. Reflects micro-instability in vocal loudness and breathiness due to stress.",
    "rms_mean": "Root-Mean-Square energy. Measures vocal intensity (loudness). Spikes can indicate tension or over-emphasis.",
    "pause_ratio": "Fraction of silence/pauses. Hesitant speakers or those under high cognitive load often pause more frequently.",
    "mfcc_1": "Mel-Frequency Cepstral Coefficient 1. Represents overall vocal tract structure.",
    "mfcc_2": "Mel-Frequency Cepstral Coefficient 2. Captures changes in resonance and pitch envelope."
}


def generate_mock_audio(scenario_type: str) -> bytes:
    """Generate a mock sine wave audio representing a speaker's voice."""
    import wave
    duration = 4.0
    sample_rate = 22050
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples)
    
    # Generate some modulation to sound slightly more interesting than a flat sine wave
    if not scenario_type or "truth" in scenario_type.lower() or "calm" in scenario_type.lower():
        # Calm tone: steady pitch (say, 150 Hz to mimic male voice)
        frequency = 150.0
        modulation = 1.0 + 0.05 * np.sin(2 * np.pi * 0.5 * t)
    else:
        # Stressed/nervous tone: fluctuating pitch
        frequency = 220.0
        # Highly modulated pitch representing shaky voice
        modulation = 1.0 + 0.2 * np.sin(2 * np.pi * 4.0 * t)
        
    audio_freq = frequency * modulation
    audio_phase = 2 * np.pi * np.cumsum(audio_freq) / sample_rate
    audio = (np.sin(audio_phase) * 32767 * 0.4).astype(np.int16)
    
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


def render_highlighted_text(text: str, important_words: list) -> str:
    """Render HTML with words highlighted based on LIME weights."""
    import re
    if not important_words:
        return f"<div style='font-size: 1.15rem; line-height: 1.8;'>{text}</div>"
        
    word_map = {}
    for item in important_words:
        w_clean = item["word"].lower().strip('.,!?()[]"\'')
        word_map[w_clean] = (item["weight"], item["direction"])
        
    tokens = re.split(r'(\s+|[.,!?()\[\]"\'])', text)
    
    html_tokens = []
    for token in tokens:
        token_clean = token.lower().strip('.,!?()[]"\'')
        if token_clean in word_map:
            weight, direction = word_map[token_clean]
            opacity = min(0.6, 0.15 + (abs(weight) / 0.25) * 0.45)
            
            if direction == "lie":
                bg_color = f"rgba(255, 71, 87, {opacity:.2f})"
                text_color = "#FF8F9A"
                border_color = "rgba(255, 71, 87, 0.4)"
                desc = f"Lie Weight: {weight:+.4f}"
            else:
                bg_color = f"rgba(46, 204, 113, {opacity:.2f})"
                text_color = "#8EE0A5"
                border_color = "rgba(46, 204, 113, 0.4)"
                desc = f"Truth Weight: {weight:+.4f}"
                
            tooltip = f"word='{token}' | {desc}"
            html_tokens.append(
                f'<span style="background-color: {bg_color}; color: {text_color}; '
                f'padding: 2px 6px; border-radius: 6px; border: 1px solid {border_color}; '
                f'cursor: help; margin: 0 1px; font-weight: 500;" '
                f'title="{tooltip}">{token}</span>'
            )
        else:
            html_tokens.append(token)
            
    joined_html = "".join(html_tokens)
    return f"""
    <div class="card" style="padding: 20px; border-left: 5px solid #6C63FF; background: rgba(255,255,255,0.02);">
        <div style="font-size: 0.8rem; color: #888; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.8px;">
            Highlighted Statement (Hover for weights)
        </div>
        <div style="font-size: 1.15rem; line-height: 1.8; color: #F0F0F0; font-family: \'Inter\', sans-serif;">
            {joined_html}
        </div>
    </div>
    """


# ──────────────────────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = True


# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0;">
        <h2 style="color:#6C63FF; margin:0; font-weight:700;">AI Lie Detector</h2>
        <p style="color:#888; font-size:0.8rem; margin-top:4px;">v1.0.0 — Probabilistic Estimation</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Demo mode toggle
    st.session_state.demo_mode = st.toggle(
        "Demo Mode (no API required)",
        value=st.session_state.demo_mode,
        help="Demo mode uses mock predictions. Disable to connect to live API.",
    )

    if not st.session_state.demo_mode:
        api_url = st.text_input("API URL", value=API_BASE_URL)
        API_BASE_URL = api_url

    st.divider()

    # Model selection
    st.markdown("**Model Settings**")
    nlp_model = st.selectbox("NLP Model", ["roberta", "bert", "deberta", "baseline"], index=0)
    fusion_method = st.selectbox("Fusion Method", ["hybrid", "late", "early"], index=0)
    include_explanation = st.checkbox("Include Explanation", value=True)

    st.divider()

    # Analytics summary
    st.markdown("**Session Analytics**")
    total = len(st.session_state.history)
    lies = sum(1 for h in st.session_state.history if "lie" in h.get("verdict", ""))
    truths = total - lies

    col1, col2 = st.columns(2)
    col1.metric("Total Analyzed", total)
    col2.metric("Deception Flags", lies)

    if total > 0:
        st.progress(lies / total if total > 0 else 0, text=f"Lie rate: {lies/total*100:.0f}%")

    st.divider()

    # Ethical disclaimer
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <b>Ethical Notice</b><br>
    This system provides <em>probabilistic estimates</em> based on statistical patterns.
    Results are <strong>NOT</strong> admissible as evidence and should never be used
    for legal, criminal, or employment decisions.
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Main Content
# ──────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center; padding: 40px 0 20px;">
    <h1 style="font-size:2.8rem; font-weight:800; background:linear-gradient(135deg,#6C63FF,#FF6584);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:8px;">
        AI-Powered Lie Detection System
    </h1>
    <p style="color:#888; font-size:1.1rem; max-width:600px; margin:0 auto;">
        Multimodal deception likelihood estimation using NLP + Voice Analysis
    </p>
</div>
""", unsafe_allow_html=True)

# ── Tab Navigation ────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Text Analysis",
    "Audio Analysis",
    "Multimodal Analysis",
    "Analytics Dashboard",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1: Text Analysis
# ═══════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<div class="section-header">Text Statement Analysis</div>', unsafe_allow_html=True)
    st.caption("Enter a text statement to analyze for deception-correlated linguistic patterns.")

    # Preset Quick Samples
    st.markdown("<p style='font-size: 0.9rem; color: #AAA; margin-bottom: 4px;'><b>Quick Fill Preset Statements:</b></p>", unsafe_allow_html=True)
    
    samples = [
        ("Routine (Truth)", "I went to the grocery store around 3pm, bought some milk and bread, and came back home by 4."),
        ("Minor Fault (Truth)", "I forgot to submit the report on Friday because I had a doctor's appointment and lost track of time."),
        ("The Alibi (Deception)", "I was definitely at home the entire night and never left, not even once. You can ask my dog."),
        ("The Denial (Deception)", "I have absolutely no idea how that money disappeared from the account, I swear I never touched it."),
        ("Evasion (Deception)", "I categorically did not send that email to the competitor, I wasn't even near my computer at that time.")
    ]
    
    cols_s = st.columns(len(samples))
    for idx, (label, val) in enumerate(samples):
        if cols_s[idx].button(label, key=f"btn_sample_t_{idx}", use_container_width=True):
            st.session_state.text_input = val
            st.rerun()

    text_input = st.text_area(
        "Enter statement to analyze:",
        value=st.session_state.get("text_input", ""),
        placeholder="e.g., I was at home the entire evening and did not leave at any point...",
        height=140,
        key="text_input_area",
        label_visibility="collapsed",
    )
    st.session_state.text_input = text_input

    col_btn1, col_btn2, _ = st.columns([1.2, 1.2, 4])
    analyze_text_btn = col_btn1.button("Analyze Text", use_container_width=True, key="btn_analyze_text")
    
    def reset_text_input():
        st.session_state.text_input = ""
    clear_btn = col_btn2.button("Clear", use_container_width=True, key="btn_clear_text", on_click=reset_text_input)

    if clear_btn:
        st.rerun()

    if analyze_text_btn and text_input.strip():
        with st.spinner("Analyzing text patterns..."):
            time.sleep(0.5)  # UX pause

            if st.session_state.demo_mode:
                result = _mock_text_response(text_input)
            else:
                result = make_api_call(
                    "predict/text",
                    json={"text": text_input, "model": nlp_model, "include_explanation": include_explanation},
                )
                if result is None:
                    result = _mock_text_response(text_input)

        if result:
            # Add to history
            st.session_state.history.append({
                "type": "text", "input": text_input[:80],
                "verdict": result.get("verdict", ""), "confidence": result.get("confidence", 0)
            })

            st.markdown("---")
            st.markdown("**Analysis Results**")

            # Verdict + Gauge
            col_v, col_g = st.columns([1, 1])

            with col_v:
                st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
                verdict = result.get("verdict", "uncertain")
                st.markdown(render_verdict_badge(verdict), unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                col_m1, col_m2 = st.columns(2)
                col_m1.metric(
                    "Lie Probability",
                    f"{result.get('lie_probability', 0)*100:.1f}%",
                    delta=f"{(result.get('lie_probability', 0.5) - 0.5)*100:+.1f}% vs baseline",
                    delta_color="inverse",
                )
                col_m2.metric(
                    "Truth Probability",
                    f"{result.get('truth_probability', 0)*100:.1f}%",
                )

                st.caption(f"Model: `{result.get('model_used', 'unknown')}` | "
                           f"Time: `{result.get('processing_time_ms', 0):.0f}ms`")
                st.markdown('</div>', unsafe_allow_html=True)

            with col_g:
                gauge_fig = render_confidence_gauge(
                    result.get("confidence", 0.5),
                    result.get("lie_probability", 0.5),
                    "Lie Probability Meter"
                )
                st.plotly_chart(gauge_fig, use_container_width=True)

            # Word importance highlights
            if include_explanation and result.get("explanation"):
                explanation = result["explanation"]
                if explanation.get("important_words"):
                    highlight_html = render_highlighted_text(text_input, explanation["important_words"])
                    st.markdown(highlight_html, unsafe_allow_html=True)

            # Word importance explanation chart
            if include_explanation and result.get("explanation"):
                explanation = result["explanation"]
                if explanation.get("important_words"):
                    st.markdown("**Explainability — Word Contributions**")
                    word_fig = render_word_importance_chart(explanation["important_words"])
                    if word_fig:
                        st.plotly_chart(word_fig, use_container_width=True)

                    with st.expander("Detailed Word Importances"):
                        for w in explanation["important_words"]:
                            dot_color = "#FF4757" if w["direction"] == "lie" else "#2ECC71"
                            st.markdown(
                                f"<span style='color:{dot_color}; font-size:1.1rem; margin-right:6px;'>●</span> "
                                f"**{w['word']}** — weight: {w['weight']:+.4f} ({w['direction']})",
                                unsafe_allow_html=True
                            )

    elif analyze_text_btn and not text_input.strip():
        st.warning("Please enter a text statement to analyze.")


# ═══════════════════════════════════════════════════════════════
# TAB 2: Audio Analysis
# ═══════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="section-header">Voice Recording Analysis</div>', unsafe_allow_html=True)
    st.caption("Upload an audio file, record directly from your microphone, or choose a preset demo scenario.")

    audio_input_method = st.radio(
        "Input method:", ["Upload Audio File", "Record with Microphone", "Quick Load Demo Scenario"],
        horizontal=True, label_visibility="collapsed",
        key="audio_input_method_select"
    )

    audio_file = None
    audio_duration = 5.0

    if audio_input_method == "Quick Load Demo Scenario":
        st.markdown("<p style='font-size: 0.9rem; color: #AAA; margin-bottom: 4px;'><b>Select preset audio scenario:</b></p>", unsafe_allow_html=True)
        audio_scenario = st.selectbox(
            "Scenario:",
            [
                "Routine Statement (Expected: Calm / Truthful)",
                "The Alibi Statement (Expected: Stressed / Deceptive)",
                "The Denial Statement (Expected: Stressed / Deceptive)"
            ],
            key="audio_preset_scenario"
        )
        scenario_key = "calm" if "routine" in audio_scenario.lower() else "stressed"
        mock_wav_bytes = generate_mock_audio(scenario_key)
        audio_file = io.BytesIO(mock_wav_bytes)
        audio_duration = 4.0
        
        st.audio(mock_wav_bytes, format="audio/wav")
        st.info(f"Demo Audio Loaded: {audio_scenario} (4.0s, 22050 Hz)")

    elif audio_input_method == "Upload Audio File":
        uploaded = st.file_uploader(
            "Drop audio file here", type=["wav", "mp3", "ogg", "flac", "m4a"],
            key="audio_uploader", label_visibility="collapsed",
        )
        if uploaded:
            st.audio(uploaded, format=f"audio/{uploaded.type.split('/')[-1]}")
            audio_file = uploaded
            try:
                import soundfile as sf
                data, sr = sf.read(io.BytesIO(uploaded.read()))
                audio_duration = len(data) / sr
                uploaded.seek(0)
                st.info(f"Duration: {audio_duration:.1f}s | Sample Rate: {sr} Hz")
            except Exception:
                audio_duration = 5.0

    else:
        try:
            from audiorecorder import audiorecorder
            audio = audiorecorder("Start Recording", "Stop Recording")
            if len(audio) > 0:
                st.audio(audio.export().read(), format="audio/wav")
                audio_duration = len(audio) / 1000
                audio_bytes = io.BytesIO()
                audio.export(audio_bytes, format="wav")
                audio_file = audio_bytes
        except ImportError:
            st.info("Install `streamlit-audiorecorder` for mic recording: `pip install streamlit-audiorecorder`")
            st.markdown("**Demo audio files:**")
            for sample in ["sample_truth.wav", "sample_lie.wav"]:
                st.code(f"# Load from: data/audio/samples/{sample}")

    analyze_audio_btn = st.button("Analyze Audio", use_container_width=False, key="btn_analyze_audio")

    if analyze_audio_btn:
        with st.spinner("Extracting audio features..."):
            time.sleep(0.8)

            if st.session_state.demo_mode or audio_file is None:
                result = _mock_audio_response(audio_duration)
            else:
                if hasattr(audio_file, "seek"):
                    audio_file.seek(0)
                result = make_api_call(
                    "predict/audio",
                    files={"file": ("audio.wav", audio_file, "audio/wav")},
                    data={"include_features": "true", "include_explanation": str(include_explanation).lower()},
                )
                if result is None:
                    result = _mock_audio_response(audio_duration)

        if result:
            st.session_state.history.append({
                "type": "audio", "input": "Audio recording",
                "verdict": result.get("voice_prediction", ""), "confidence": result.get("confidence", 0)
            })

            st.markdown("---")
            st.markdown("**Audio Analysis Results**")

            col_v, col_g = st.columns([1, 1])
            with col_v:
                st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
                voice_pred = result.get("voice_prediction", "no_stress")
                stress_prob = result.get("stress_probability", 0.5)

                if voice_pred == "stress_detected":
                    st.markdown('<div class="verdict-lie">STRESS DETECTED</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="verdict-truth">NO SIGNIFICANT STRESS</div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                c1.metric("Stress Probability", f"{stress_prob*100:.1f}%")
                c2.metric("Duration", f"{result.get('audio_duration_seconds', 0):.1f}s")
                st.caption(f"Model: `{result.get('model_used', 'rf')}` | Time: `{result.get('processing_time_ms', 0):.0f}ms`")
                st.markdown('</div>', unsafe_allow_html=True)

            with col_g:
                gauge = render_confidence_gauge(
                    result.get("confidence", 0.5),
                    stress_prob,
                    "Stress Level Meter"
                )
                st.plotly_chart(gauge, use_container_width=True)

            # Audio feature breakdown
            if result.get("audio_features"):
                st.markdown("**Extracted Audio Features**")
                features = result["audio_features"]
                
                # Educational descriptions expander
                with st.expander("Learn what these acoustic metrics mean"):
                    for key, desc in FEATURE_DESCRIPTIONS.items():
                        st.markdown(f"**{key.replace('_', ' ').title()}**: {desc}")
                
                feat_cols = st.columns(min(4, len(features)))
                for i, (feat_name, val) in enumerate(list(features.items())[:8]):
                    with feat_cols[i % 4]:
                        desc_text = FEATURE_DESCRIPTIONS.get(feat_name.lower(), "Vocal resonance attribute.")
                        st.metric(
                            feat_name.replace("_", " ").title(), 
                            f"{val:.4f}",
                            help=desc_text
                        )


# ═══════════════════════════════════════════════════════════════
# TAB 3: Multimodal Analysis
# ═══════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-header">Multimodal Deception Analysis</div>', unsafe_allow_html=True)
    st.caption("Combine text + voice analysis for the most accurate deception likelihood estimate.")

    # Multimodal Preset Scenarios
    st.markdown("<p style='font-size: 0.9rem; color: #AAA; margin-bottom: 4px;'><b>Quick Load Multimodal Scenarios:</b></p>", unsafe_allow_html=True)
    
    mm_scenarios = {
        "-- Custom inputs --": {
            "text": "",
            "audio_type": None
        },
        "Scenario 1: Admitting Minor Fault (Expected: Truth / Calm)": {
            "text": "I forgot to submit the report on Friday because I had a doctor's appointment in the afternoon and completely lost track of time.",
            "audio_type": "calm"
        },
        "Scenario 2: The Evasion Alibi (Expected: Lie / Stressed)": {
            "text": "I was definitely at home the entire night and never left, not even once. You can ask my dog.",
            "audio_type": "stressed"
        },
        "Scenario 3: Direct Explanation (Expected: Truth / Calm)": {
            "text": "I arrived late to the office because the train was delayed by about 20 minutes at the Central station.",
            "audio_type": "calm"
        },
        "Scenario 4: Over-Justification (Expected: Lie / Stressed)": {
            "text": "I am always completely honest. Everyone knows I am the most trustworthy person in this office, so I wouldn't do that.",
            "audio_type": "stressed"
        }
    }
    
    mm_preset_selection = st.selectbox(
        "Select scenario to pre-fill:",
        list(mm_scenarios.keys()),
        index=0,
        key="mm_scenario_select"
    )
    
    if mm_preset_selection != "-- Custom inputs --":
        st.session_state.mm_text = mm_scenarios[mm_preset_selection]["text"]
        
    mm_col1, mm_col2 = st.columns([1, 1])

    with mm_col1:
        st.markdown("**Text Statement**")
        mm_text = st.text_area(
            "Text:", 
            value=st.session_state.get("mm_text", ""),
            placeholder="Enter the statement spoken in the audio recording...",
            height=120, key="mm_text_area", label_visibility="collapsed",
        )
        st.session_state.mm_text = mm_text

    with mm_col2:
        st.markdown("**Audio Recording**")
        mm_audio_file = None
        if mm_preset_selection != "-- Custom inputs --":
            audio_type = mm_scenarios[mm_preset_selection]["audio_type"]
            mock_wav_bytes = generate_mock_audio(audio_type)
            mm_audio_file = io.BytesIO(mock_wav_bytes)
            st.audio(mock_wav_bytes, format="audio/wav")
            st.info("Preset Audio Loaded.")
        else:
            mm_audio = st.file_uploader(
                "Audio:", type=["wav", "mp3", "ogg", "flac"], key="mm_audio",
                label_visibility="collapsed",
            )
            if mm_audio:
                st.audio(mm_audio, format=f"audio/{mm_audio.type.split('/')[-1]}")
                mm_audio_file = mm_audio

    col_mm_btn, _ = st.columns([2, 6])
    analyze_mm_btn = col_mm_btn.button("Run Multimodal Analysis", use_container_width=True, key="btn_analyze_mm")

    if analyze_mm_btn:
        if not mm_text.strip():
            st.warning("Please enter the text statement.")
        elif mm_audio_file is None:
            st.warning("Please upload an audio file or select a preset scenario.")
        else:
            with st.spinner("Running multimodal analysis..."):
                time.sleep(1.0)

                if st.session_state.demo_mode:
                    result = _mock_multimodal_response(mm_text)
                else:
                    if hasattr(mm_audio_file, "seek"):
                        mm_audio_file.seek(0)
                    result = make_api_call(
                        "predict/multimodal",
                        files={"file": ("audio.wav", mm_audio_file, "audio/wav")},
                        data={
                            "text": mm_text, "fusion_method": fusion_method,
                            "include_explanation": str(include_explanation).lower(),
                        },
                    )
                    if result is None:
                        result = _mock_multimodal_response(mm_text)

            if result:
                st.session_state.history.append({
                    "type": "multimodal", "input": mm_text[:80],
                    "verdict": result.get("final_prediction", ""), "confidence": result.get("confidence", 0)
                })

                st.markdown("---")
                st.markdown("**Multimodal Fusion Results**")

                # Top-level verdict
                final_verdict = result.get("final_prediction", "uncertain")
                confidence = result.get("confidence", 0.5)
                lie_prob = result.get("lie_probability", 0.5)

                result_col1, result_col2, result_col3 = st.columns([2, 2, 2])

                with result_col1:
                    st.markdown('<div class="card fade-in" style="text-align:center;">', unsafe_allow_html=True)
                    st.markdown(render_verdict_badge(final_verdict), unsafe_allow_html=True)
                    st.markdown(f"""
                    <br>
                    <div class="metric-value">{lie_prob*100:.1f}%</div>
                    <div class="metric-label">Lie Probability</div>
                    """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with result_col2:
                    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
                    st.markdown("**Text Analysis**")
                    text_lie = result.get("text_lie_probability", 0.5)
                    text_badge = "Deceptive" if text_lie > 0.5 else "Truthful"
                    st.markdown(f"Prediction: {text_badge}")
                    st.progress(text_lie, text=f"Lie probability: {text_lie*100:.1f}%")

                    st.markdown("**Audio Analysis**")
                    audio_stress = result.get("audio_stress_probability", 0.5)
                    audio_badge = "Stressed" if audio_stress > 0.5 else "Calm"
                    st.markdown(f"Prediction: {audio_badge}")
                    st.progress(audio_stress, text=f"Stress: {audio_stress*100:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)

                with result_col3:
                    gauge = render_confidence_gauge(confidence, lie_prob, "Final Lie Score")
                    st.plotly_chart(gauge, use_container_width=True)

                # Highlighted statement visualizer
                if include_explanation and result.get("explanation"):
                    exp = result["explanation"]
                    if exp.get("important_words"):
                        highlight_html = render_highlighted_text(mm_text, exp["important_words"])
                        st.markdown(highlight_html, unsafe_allow_html=True)

                # Explainability charts
                if include_explanation and result.get("explanation"):
                    exp = result["explanation"]
                    ex_col1, ex_col2 = st.columns(2)

                    with ex_col1:
                        if exp.get("important_words"):
                            word_fig = render_word_importance_chart(exp["important_words"])
                            if word_fig:
                                st.plotly_chart(word_fig, use_container_width=True)

                    with ex_col2:
                        if exp.get("audio_features"):
                            audio_fig = render_audio_feature_chart(exp["audio_features"])
                            if audio_fig:
                                st.plotly_chart(audio_fig, use_container_width=True)

                # JSON output
                with st.expander("Raw API Response"):
                    st.json(result)


# ═══════════════════════════════════════════════════════════════
# TAB 4: Analytics Dashboard
# ═══════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<div class="section-header">Session Analytics Dashboard</div>', unsafe_allow_html=True)

    history = st.session_state.history

    if not history:
        st.info("No analyses yet. Run some predictions to see analytics here.")
    else:
        # Summary metrics
        total = len(history)
        text_count = sum(1 for h in history if h["type"] == "text")
        audio_count = sum(1 for h in history if h["type"] == "audio")
        mm_count = sum(1 for h in history if h["type"] == "multimodal")
        lie_flags = sum(1 for h in history if "lie" in h.get("verdict", "").lower())
        avg_conf = np.mean([h.get("confidence", 0.5) for h in history])

        cols = st.columns(5)
        cols[0].metric("Total Analyses", total)
        cols[1].metric("Text", text_count)
        cols[2].metric("Audio", audio_count)
        cols[3].metric("Multimodal", mm_count)
        cols[4].metric("Deception Flags", lie_flags)

        st.markdown("---")
        dash_col1, dash_col2 = st.columns(2)

        # Verdict distribution pie chart
        with dash_col1:
            verdicts = [h.get("verdict", "uncertain") for h in history]
            verdict_counts = {}
            for v in verdicts:
                verdict_counts[v] = verdict_counts.get(v, 0) + 1

            fig_pie = go.Figure(go.Pie(
                labels=list(verdict_counts.keys()),
                values=list(verdict_counts.values()),
                hole=0.5,
                marker=dict(colors=["#FF4757", "#FFA500", "#2ECC71", "#3498DB"]),
                textfont={"color": "white"},
            ))
            fig_pie.update_layout(
                title="Verdict Distribution",
                title_font={"color": "#E0E0E0", "size": 14},
                paper_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter", "color": "#E0E0E0"},
                legend={"font": {"color": "#E0E0E0"}},
                height=300, margin={"t": 40, "b": 10},
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Confidence over time line chart
        with dash_col2:
            confs = [h.get("confidence", 0.5) * 100 for h in history]
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                y=confs, mode="lines+markers",
                line={"color": "#6C63FF", "width": 2.5},
                marker={"color": "#FF6584", "size": 8},
                fill="tozeroy",
                fillcolor="rgba(108,99,255,0.1)",
                name="Confidence",
            ))
            fig_line.update_layout(
                title="Confidence Over Analyses",
                title_font={"color": "#E0E0E0", "size": 14},
                xaxis={"title": "Analysis #", "color": "#888"},
                yaxis={"title": "Confidence (%)", "color": "#888", "range": [0, 105]},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.02)",
                font={"family": "Inter", "color": "#E0E0E0"},
                height=300, margin={"t": 40, "b": 40},
            )
            st.plotly_chart(fig_line, use_container_width=True)

        # History table
        st.markdown("**Analysis History**")
        table_data = []
        for i, h in enumerate(history):
            table_data.append({
                "#": i + 1,
                "Type": h["type"].title(),
                "Input Preview": h.get("input", "")[:50] + "..." if len(h.get("input", "")) > 50 else h.get("input", ""),
                "Verdict": h.get("verdict", "unknown"),
                "Confidence": f"{h.get('confidence', 0)*100:.1f}%",
            })

        import pandas as pd
        df_history = pd.DataFrame(table_data)
        st.dataframe(
            df_history,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Verdict": st.column_config.TextColumn("Verdict"),
                "Confidence": st.column_config.TextColumn("Confidence"),
            },
        )

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
