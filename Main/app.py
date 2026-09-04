"""
Phishing email classifier — demo interface.
Loads Aetheris-Lite (SGD) and NexusVector-X (XGBoost), both wrapped in a
fixed-threshold classifier, and lets a visitor test either model on a
pasted email.
"""

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, ClassifierMixin

# ----------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Phishing Email Classifier",
    page_icon="◆",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------
# Model wrapper — must match the class used when the models were saved,
# or joblib.load() will fail to unpickle them.
# ----------------------------------------------------------------------
class CustomThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def predict(self, X):
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


REQUIRED_COLUMNS = ["text", "has_attachments", "url_count", "urls"]

URL_PATTERN = re.compile(r'(https?://[^\s<>"\')\]]+|www\.[^\s<>"\')\]]+)', re.IGNORECASE)

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILES = {
    "Lite": {"path": BASE_DIR / "Aetheris-Lite.joblib", "family": "SGD, linear"},
    "NexusVector-X": {"path": BASE_DIR / "NexusVector-X.joblib", "family": "XGBoost, gradient-boosted trees"},
}

LOW_CONFIDENCE_THRESHOLD = 0.60

DISCLAIMER = (
    "The displayed accuracy represents the model's confidence, not a perfect "
    "or guaranteed accuracy you can rely on. If you don't have a label to "
    "compare the prediction against, please don't treat the model's output "
    "as statistical fact — treat it as support for an analytical decision."
)

USAGE_TIP = (
    "◆ <b>NexusVector-X</b> is stronger at catching phishing — pick it when you suspect spam. "
    "◆ <b>Lite</b> is stronger at recognizing legitimate mail — pick it when you suspect the email is actually important."
)


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)


def extract_urls(text: str):
    """Mirrors the training data convention: join every URL found in the
    text with a single space, or return the literal 'No URL' sentinel."""
    found = URL_PATTERN.findall(text)
    if not found:
        return "No URL", 0
    return " ".join(found), len(found)


def validate_input_frame(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    return (not missing and not extra), missing, extra


# ----------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

    :root {
        --bg: #F5F6F4;
        --surface: #FFFFFF;
        --line: #DDE3DD;
        --ink: #1C2321;
        --ink-soft: #5B6660;
        --accent: #3D6B5C;
        --accent-soft: #E4EBE6;
        --danger: #A6423A;
        --danger-soft: #F3E6E4;
    }

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background: var(--bg); color: var(--ink); }

    #MainMenu, header, footer { visibility: hidden; }
    .block-container { padding-top: 3rem; padding-bottom: 4rem; max-width: 700px; }

    h1, h2, h3 { font-family: 'Sora', sans-serif; color: var(--ink); }

    .app-title { font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.6rem; margin-bottom: 0.15rem; }
    .app-subtitle { color: var(--ink-soft); font-size: 0.95rem; margin-bottom: 2rem; }

    /* Input panel */
    .input-panel {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 0;
        overflow: hidden;
        margin-bottom: 1.5rem;
    }
    .panel-zone-text { padding: 1.1rem 1.1rem 0.4rem 1.1rem; }
    .panel-divider { border-top: 1px solid var(--line); margin: 0; }
    .panel-zone-controls { padding: 1rem 1.1rem 1.1rem 1.1rem; }

    .stTextArea textarea {
        border: none !important;
        background: transparent !important;
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.95rem;
        color: var(--ink);
        box-shadow: none !important;
        resize: vertical;
    }
    .stTextArea textarea:focus { outline: none !important; box-shadow: none !important; }

    .field-label {
        font-size: 0.78rem;
        color: var(--ink-soft);
        margin-bottom: 0.3rem;
        font-weight: 500;
    }

    /* Segmented controls (has_attachments / url mode / model) */
    div[data-testid="stSegmentedControl"] button {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.85rem !important;
        border-color: var(--line) !important;
        color: var(--ink-soft) !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
        background: var(--accent) !important;
        border-color: var(--accent) !important;
        color: #FFFFFF !important;
    }

    .stNumberInput input, .stTextArea textarea, .stTextInput input {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Primary button */
    .stButton button {
        background: var(--accent);
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        font-family: 'Sora', sans-serif;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0.55rem 1.6rem;
    }
    .stButton button:hover { background: #335A4D; color: #FFFFFF; }

    /* Result strip */
    .result-strip {
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 1.2rem 1.3rem;
        background: var(--surface);
    }
    .result-strip.is-phishing { border-color: var(--danger); background: var(--danger-soft); }
    .result-strip.is-legit { border-color: var(--accent); background: var(--accent-soft); }

    .result-label {
        font-family: 'Sora', sans-serif;
        font-weight: 700;
        font-size: 1.25rem;
    }
    .result-label.is-phishing { color: var(--danger); }
    .result-label.is-legit { color: var(--accent); }

    .result-meta {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        color: var(--ink-soft);
        margin-top: 0.35rem;
    }

    .disclaimer {
        font-size: 0.78rem;
        color: var(--ink-soft);
        margin-top: 0.9rem;
        line-height: 1.5;
        border-top: 1px solid var(--line);
        padding-top: 0.75rem;
    }

    .usage-tip {
        font-size: 0.76rem;
        color: var(--ink-soft);
        margin-top: 0.5rem;
        line-height: 1.6;
    }
    .usage-tip b { color: var(--ink); }

    .low-conf-banner {
        font-size: 0.82rem;
        color: var(--ink);
        background: #FBF3E3;
        border: 1px solid #E3C77F;
        border-radius: 8px;
        padding: 0.65rem 0.85rem;
        margin-bottom: 0.7rem;
        line-height: 1.5;
    }

    .disagree-banner {
        font-size: 0.82rem;
        color: var(--ink);
        background: var(--danger-soft);
        border: 1px solid var(--danger);
        border-radius: 8px;
        padding: 0.65rem 0.85rem;
        margin-bottom: 0.7rem;
        line-height: 1.5;
    }

    .second-opinion {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        color: var(--ink-soft);
        margin-top: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown('<div class="app-title">Phishing Email Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Paste an email below to see how the model reads it.</div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Input panel
# ----------------------------------------------------------------------
st.markdown('<div class="input-panel">', unsafe_allow_html=True)

st.markdown('<div class="panel-zone-text">', unsafe_allow_html=True)
email_text = st.text_area(
    "Email text",
    height=180,
    placeholder="Paste the subject and body of the email here...",
    label_visibility="collapsed",
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="panel-divider"></div>', unsafe_allow_html=True)

st.markdown('<div class="panel-zone-controls">', unsafe_allow_html=True)

row1_a, row1_b = st.columns(2)
with row1_a:
    st.markdown('<div class="field-label">Attachment</div>', unsafe_allow_html=True)
    has_attachment_choice = st.segmented_control(
        "Attachment", options=["No", "Yes"], default="No", label_visibility="collapsed"
    )
with row1_b:
    st.markdown('<div class="field-label">Links</div>', unsafe_allow_html=True)
    url_mode = st.segmented_control(
        "Links", options=["Automatic", "Manual"], default="Automatic", label_visibility="collapsed"
    )

manual_url_count = None
manual_urls_text = ""
if url_mode == "Manual":
    st.markdown('<div class="field-label" style="margin-top:0.7rem;">Manual link details</div>', unsafe_allow_html=True)
    m_a, m_b = st.columns([1, 2])
    with m_a:
        manual_url_count = st.number_input(
            "Link count", min_value=0, value=1, step=1, label_visibility="collapsed"
        )
    with m_b:
        manual_urls_text = st.text_input(
            "Links", placeholder="paste the link(s), space-separated", label_visibility="collapsed"
        )

st.markdown('<div class="field-label" style="margin-top:0.9rem;">Model</div>', unsafe_allow_html=True)
model_choice = st.segmented_control(
    "Model", options=["Lite", "NexusVector-X"], default="Lite", label_visibility="collapsed"
)
st.markdown(f'<div class="usage-tip">{USAGE_TIP}</div>', unsafe_allow_html=True)

st.markdown('<div style="margin-top:0.9rem;">', unsafe_allow_html=True)
analyze_clicked = st.button("Analyze email", use_container_width=False)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # panel-zone-controls
st.markdown("</div>", unsafe_allow_html=True)  # input-panel

# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
if analyze_clicked:
    if not email_text or not email_text.strip():
        st.error("Paste an email's text before running the analysis.")
        st.stop()

    has_attachments = 1 if has_attachment_choice == "Yes" else 0

    if url_mode == "Automatic":
        urls_value, url_count_value = extract_urls(email_text)
    else:
        typed = (manual_urls_text or "").strip()
        if typed:
            urls_value, url_count_value = typed, int(manual_url_count or 0)
        else:
            urls_value, url_count_value = "No URL", 0

    input_df = pd.DataFrame(
        [{
            "text": email_text,
            "has_attachments": has_attachments,
            "url_count": url_count_value,
            "urls": urls_value,
        }]
    )[REQUIRED_COLUMNS]

    is_valid, missing, extra = validate_input_frame(input_df)
    if not is_valid:
        st.error(
            "The data sent to the model doesn't match what it expects.\n\n"
            f"Required columns: {REQUIRED_COLUMNS}\n"
            f"Missing: {missing if missing else 'none'}\n"
            f"Unexpected: {extra if extra else 'none'}\n\n"
            "Fix the column names and try again."
        )
        st.stop()

    def run_model(model_key):
        info = MODEL_FILES[model_key]
        clf = load_model(info["path"])
        p = clf.predict_proba(input_df)[:, 1][0]
        pred = int(p >= clf.threshold)
        # Confidence in the ACTUAL predicted class, not raw phishing-probability.
        # Always in [0.5, 1.0]: how sure the model is of the label it gave.
        conf = p if pred == 1 else (1 - p)
        return {
            "key": model_key,
            "family": info["family"],
            "prediction": pred,
            "label": "Phishing" if pred == 1 else "Legitimate",
            "confidence": conf,
        }

    with st.spinner("Reading the email..."):
        primary = run_model(model_choice)
        secondary = None
        if primary["confidence"] < LOW_CONFIDENCE_THRESHOLD:
            other_key = "NexusVector-X" if model_choice == "Lite" else "Lite"
            secondary = run_model(other_key)

    css_class = "is-phishing" if primary["prediction"] == 1 else "is-legit"

    # Banner shown only when the primary model wasn't confident.
    if secondary is not None:
        if secondary["prediction"] == primary["prediction"]:
            st.markdown(
                f'<div class="low-conf-banner">⚠ {primary["key"]} wasn\'t very sure about this one '
                f'({primary["confidence"]*100:.0f}% confidence). {secondary["key"]} was consulted '
                f'automatically and agrees: <b>{secondary["label"]}</b> '
                f'({secondary["confidence"]*100:.0f}% confidence). Treat this as a lean, not a fact — '
                f'take a closer look yourself.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="disagree-banner">⚠ The two models disagree on this one. '
                f'{primary["key"]} says <b>{primary["label"]}</b> ({primary["confidence"]*100:.0f}% confidence), '
                f'{secondary["key"]} says <b>{secondary["label"]}</b> ({secondary["confidence"]*100:.0f}% confidence). '
                f'Neither result should be trusted here — check the email yourself before deciding.</div>',
                unsafe_allow_html=True,
            )

    st.markdown(f'<div class="result-strip {css_class}">', unsafe_allow_html=True)
    st.markdown(f'<div class="result-label {css_class}">{primary["label"]}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="result-meta">confidence {primary["confidence"] * 100:.1f}% &nbsp;·&nbsp; '
        f'model: {primary["key"]} ({primary["family"]})</div>',
        unsafe_allow_html=True,
    )
    if secondary is not None:
        st.markdown(
            f'<div class="second-opinion">second opinion — {secondary["key"]}: '
            f'{secondary["label"]} ({secondary["confidence"]*100:.1f}%)</div>',
            unsafe_allow_html=True,
        )
    st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
