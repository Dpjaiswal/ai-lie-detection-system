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
    page_icon="🔍",
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
        background: linear-gradient(135deg, #0D0D1A 0%, #1A1A2E 50%, #16213E 100%);
        min-height: 100vh;
    }
    .main .block-container { max-width: 1200px; padding: 2rem; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12122A 0%, #0D0D1A 100%);
        border-right: 1px solid #2C2C54;
    }

    /* ── Cards ── */
    .card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(108,99,255,0.25);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 30px rgba(0,0,0,0.3);
        transition: border-color 0.3s ease;
    }
    .card:hover { border-color: rgba(108,99,255,0.6); }

    /* ── Verdict badges ── */
    .verdict-lie {
        display: inline-block;
        background: linear-gradient(135deg, #FF4757, #FF6B81);
        color: white; border-radius: 100px;
        padding: 8px 24px; font-size: 1.1rem;
        font-weight: 700; letter-spacing: 0.5px;
        box-shadow: 0 0 20px rgba(255,71,87,0.5);
        animation: pulse 2s infinite;
    }
    .verdict-truth {
        display: inline-block;
        background: linear-gradient(135deg, #2ECC71, #27AE60);
        color: white; border-radius: 100px;
        padding: 8px 24px; font-size: 1.1rem;
        font-weight: 700; letter-spacing: 0.5px;
        box-shadow: 0 0 20px rgba(46,204,113,0.5);
    }
    .verdict-uncertain {
        display: inline-block;
        background: linear-gradient(135deg, #F39C12, #E67E22);
        color: white; border-radius: 100px;
        padding: 8px 24px; font-size: 1.1rem;
        font-weight: 700;
    }

    /* ── Pulse animation ── */
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 20px rgba(255,71,87,0.5); }
        50% { box-shadow: 0 0 35px rgba(255,71,87,0.8); }
    }

    /* ── Metric tiles ── */
    .metric-tile {
        background: rgba(108,99,255,0.1);
        border: 1px solid rgba(108,99,255,0.3);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #6C63FF; }
    .metric-label { font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 0.8px; }

    /* ── Disclaimer banner ── */
    .disclaimer {
        background: rgba(255,165,0,0.1);
        border: 1px solid rgba(255,165,0,0.4);
        border-radius: 12px;
        padding: 12px 20px;
        color: #FFA500;
        font-size: 0.85rem;
        line-height: 1.6;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.3rem; font-weight: 600;
        background: linear-gradient(135deg, #6C63FF, #FF6584);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 12px;
    }

    /* ── Inputs ── */
    .stTextArea textarea {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(108,99,255,0.4) !important;
        border-radius: 12px !important;
        color: #E0E0E0 !important;
        font-size: 0.95rem !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6C63FF, #8B7CF6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.6rem 2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(108,99,255,0.4) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(108,99,255,0.6) !important;
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
        st.error("⚠️ Cannot connect to API server. Is it running on localhost:8000?")
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
        label = "⚠️ DECEPTION LIKELY" if "likely" in verdict else "⚠️ POSSIBLY DECEPTIVE"
        return f'<div class="verdict-lie">{label}</div>'
    elif "truth" in verdict.lower():
        label = "✅ LIKELY TRUTHFUL" if "likely" in verdict else "✅ POSSIBLY TRUTHFUL"
        return f'<div class="verdict-truth">{label}</div>'
    else:
        return '<div class="verdict-uncertain">❓ UNCERTAIN</div>'


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
        <div style="font-size:3rem;">🔍</div>
        <h2 style="color:#6C63FF; margin:0; font-weight:700;">AI Lie Detector</h2>
        <p style="color:#888; font-size:0.8rem; margin-top:4px;">v1.0.0 — Probabilistic Estimation</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Demo mode toggle
    st.session_state.demo_mode = st.toggle(
        "🎭 Demo Mode (no API required)",
        value=st.session_state.demo_mode,
        help="Demo mode uses mock predictions. Disable to connect to live API.",
    )

    if not st.session_state.demo_mode:
        api_url = st.text_input("API URL", value=API_BASE_URL)
        API_BASE_URL = api_url

    st.divider()

    # Model selection
    st.markdown("**⚙️ Model Settings**")
    nlp_model = st.selectbox("NLP Model", ["roberta", "bert", "deberta", "baseline"], index=0)
    fusion_method = st.selectbox("Fusion Method", ["hybrid", "late", "early"], index=0)
    include_explanation = st.checkbox("Include Explanation", value=True)

    st.divider()

    # Analytics summary
    st.markdown("**📊 Session Analytics**")
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
    "📝 Text Analysis",
    "🎙️ Audio Analysis",
    "🔀 Multimodal Analysis",
    "📊 Analytics Dashboard",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1: Text Analysis
# ═══════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<div class="section-header">📝 Text Statement Analysis</div>', unsafe_allow_html=True)
    st.caption("Enter a text statement to analyze for deception-correlated linguistic patterns.")

    text_input = st.text_area(
        "Enter statement to analyze:",
        placeholder="e.g., I was at home the entire evening and did not leave at any point...",
        height=140,
        key="text_input",
        label_visibility="collapsed",
    )

    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    analyze_text_btn = col_btn1.button("🔍 Analyze Text", use_container_width=True)
    clear_btn = col_btn2.button("🗑️ Clear", use_container_width=True)

    if clear_btn:
        st.rerun()

    if analyze_text_btn and text_input.strip():
        with st.spinner("🤖 Analyzing text patterns..."):
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
            st.markdown("**🎯 Analysis Results**")

            # Verdict + Gauge
            col_v, col_g = st.columns([1, 1])

            with col_v:
                st.markdown('<div class="card">', unsafe_allow_html=True)
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

            # Word importance explanation
            if include_explanation and result.get("explanation"):
                explanation = result["explanation"]
                if explanation.get("important_words"):
                    st.markdown("**🧠 Explainability — Word Contributions**")
                    word_fig = render_word_importance_chart(explanation["important_words"])
                    if word_fig:
                        st.plotly_chart(word_fig, use_container_width=True)

                    with st.expander("📋 Detailed Word Importances"):
                        for w in explanation["important_words"]:
                            emoji = "🔴" if w["direction"] == "lie" else "🟢"
                            st.write(f"{emoji} **{w['word']}** — weight: {w['weight']:+.4f} ({w['direction']})")

    elif analyze_text_btn and not text_input.strip():
        st.warning("⚠️ Please enter a text statement to analyze.")


# ═══════════════════════════════════════════════════════════════
# TAB 2: Audio Analysis
# ═══════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="section-header">🎙️ Voice Recording Analysis</div>', unsafe_allow_html=True)
    st.caption("Upload an audio file or record directly from your microphone.")

    audio_input_method = st.radio(
        "Input method:", ["📁 Upload Audio File", "🎤 Record with Microphone"],
        horizontal=True, label_visibility="collapsed"
    )

    audio_file = None
    audio_duration = 5.0

    if audio_input_method == "📁 Upload Audio File":
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
                st.info(f"📊 Duration: {audio_duration:.1f}s | Sample Rate: {sr} Hz")
            except Exception:
                audio_duration = 5.0

    else:
        try:
            from audiorecorder import audiorecorder
            audio = audiorecorder("🔴 Start Recording", "⏹️ Stop Recording")
            if len(audio) > 0:
                st.audio(audio.export().read(), format="audio/wav")
                audio_duration = len(audio) / 1000
                audio_bytes = io.BytesIO()
                audio.export(audio_bytes, format="wav")
                audio_file = audio_bytes
        except ImportError:
            st.info("💡 Install `streamlit-audiorecorder` for mic recording: `pip install streamlit-audiorecorder`")
            st.markdown("**Demo audio files:**")
            for sample in ["sample_truth.wav", "sample_lie.wav"]:
                st.code(f"# Load from: data/audio/samples/{sample}")

    analyze_audio_btn = st.button("🔊 Analyze Audio", use_container_width=False)

    if analyze_audio_btn:
        with st.spinner("🔊 Extracting audio features..."):
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
            st.markdown("**🎯 Audio Analysis Results**")

            col_v, col_g = st.columns([1, 1])
            with col_v:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                voice_pred = result.get("voice_prediction", "no_stress")
                stress_prob = result.get("stress_probability", 0.5)

                if voice_pred == "stress_detected":
                    st.markdown('<div class="verdict-lie">⚡ STRESS DETECTED</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="verdict-truth">✅ NO SIGNIFICANT STRESS</div>', unsafe_allow_html=True)

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
                st.markdown("**📈 Extracted Audio Features**")
                features = result["audio_features"]
                feat_cols = st.columns(min(4, len(features)))
                for i, (feat_name, val) in enumerate(list(features.items())[:8]):
                    with feat_cols[i % 4]:
                        st.metric(feat_name.replace("_", " ").title(), f"{val:.4f}")


# ═══════════════════════════════════════════════════════════════
# TAB 3: Multimodal Analysis
# ═══════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-header">🔀 Multimodal Deception Analysis</div>', unsafe_allow_html=True)
    st.caption("Combine text + voice analysis for the most accurate deception likelihood estimate.")

    mm_col1, mm_col2 = st.columns([1, 1])

    with mm_col1:
        st.markdown("**📝 Text Statement**")
        mm_text = st.text_area(
            "Text:", placeholder="Enter the statement spoken in the audio recording...",
            height=120, key="mm_text", label_visibility="collapsed",
        )

    with mm_col2:
        st.markdown("**🎙️ Audio Recording**")
        mm_audio = st.file_uploader(
            "Audio:", type=["wav", "mp3", "ogg", "flac"], key="mm_audio",
            label_visibility="collapsed",
        )
        if mm_audio:
            st.audio(mm_audio, format=f"audio/{mm_audio.type.split('/')[-1]}")

    col_mm_btn, _ = st.columns([2, 6])
    analyze_mm_btn = col_mm_btn.button("🔀 Run Multimodal Analysis", use_container_width=True)

    if analyze_mm_btn:
        if not mm_text.strip():
            st.warning("⚠️ Please enter the text statement.")
        else:
            with st.spinner("🔀 Running multimodal analysis..."):
                time.sleep(1.0)

                if st.session_state.demo_mode or mm_audio is None:
                    result = _mock_multimodal_response(mm_text)
                else:
                    mm_audio.seek(0)
                    result = make_api_call(
                        "predict/multimodal",
                        files={"file": ("audio.wav", mm_audio, "audio/wav")},
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
                st.markdown("**🎯 Multimodal Fusion Results**")

                # Top-level verdict
                final_verdict = result.get("final_prediction", "uncertain")
                confidence = result.get("confidence", 0.5)
                lie_prob = result.get("lie_probability", 0.5)

                result_col1, result_col2, result_col3 = st.columns([2, 2, 2])

                with result_col1:
                    st.markdown('<div class="card" style="text-align:center;">', unsafe_allow_html=True)
                    st.markdown(render_verdict_badge(final_verdict), unsafe_allow_html=True)
                    st.markdown(f"""
                    <br>
                    <div class="metric-value">{lie_prob*100:.1f}%</div>
                    <div class="metric-label">Lie Probability</div>
                    """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with result_col2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("**Text Analysis**")
                    text_lie = result.get("text_lie_probability", 0.5)
                    text_badge = "🔴 Deceptive" if text_lie > 0.5 else "🟢 Truthful"
                    st.markdown(f"Prediction: {text_badge}")
                    st.progress(text_lie, text=f"Lie probability: {text_lie*100:.1f}%")

                    st.markdown("**Audio Analysis**")
                    audio_stress = result.get("audio_stress_probability", 0.5)
                    audio_badge = "⚡ Stressed" if audio_stress > 0.5 else "😌 Calm"
                    st.markdown(f"Prediction: {audio_badge}")
                    st.progress(audio_stress, text=f"Stress: {audio_stress*100:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)

                with result_col3:
                    gauge = render_confidence_gauge(confidence, lie_prob, "Final Lie Score")
                    st.plotly_chart(gauge, use_container_width=True)

                # Explainability
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
                with st.expander("📄 Raw API Response"):
                    st.json(result)


# ═══════════════════════════════════════════════════════════════
# TAB 4: Analytics Dashboard
# ═══════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<div class="section-header">📊 Session Analytics Dashboard</div>', unsafe_allow_html=True)

    history = st.session_state.history

    if not history:
        st.info("📭 No analyses yet. Run some predictions to see analytics here.")
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
        st.markdown("**📋 Analysis History**")
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

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
