"""
FoxLens — Phishing Email Classifier Demo Interface.
Loads Aetheris-Lite (SGD) and NexusVector-X (XGBoost), wrapped in a
fixed-threshold classifier, letting visitors test emails and compare models.
"""

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, ClassifierMixin

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="FoxLens — Phishing Classifier",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------
# Model wrapper
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
GITHUB_REPO_URL = "https://github.com/FaroukOuici/FoxLens/tree/main"
FIVERR_PROFILE_URL = "https://www.fiverr.com/s/61bqdEA"

DISCLAIMER = (
    "The displayed score represents the model's confidence, not absolute fact. "
    "Use it to assist your judgment, not replace human verification."
)

USAGE_TIP = (
    "<b>NexusVector-X:</b> Highly sensitive to deceptive phishing attempts. &nbsp;•&nbsp; "
    "<b>Lite:</b> Conservative classifier, better at recognizing authentic mail."
)

# ----------------------------------------------------------------------
# Welcome / Disclaimer Dialog
# ----------------------------------------------------------------------
if "disclaimer_accepted" not in st.session_state:
    st.session_state.disclaimer_accepted = False


@st.dialog("Welcome to FoxLens")
def welcome_dialog():
    st.markdown(
        f"""
        Welcome to the **FoxLens** demo.
        
        Before proceeding, please note that this is an experimental project; it is not a substitute for human analysis, statistical studies, or specialized software.
        
        The project's source code and interface are available at:  
        🔗 **[FoxLens GitHub Repository]({GITHUB_REPO_URL})**
        
        Additionally, please be aware that the data you enter passes instantly through a processing pipeline to the model for prediction, after which it is **immediately cleared and not stored** in any form.
        
        If you are a client or project owner interested in a similar AI/ML solution for your project, feel free to get in touch via my Fiverr profile:  
        💼 **[Fiverr Profile]({FIVERR_PROFILE_URL})**
        """
    )
    if st.button("Accept & Continue", type="primary", use_container_width=True):
        st.session_state.disclaimer_accepted = True
        st.rerun()


if not st.session_state.disclaimer_accepted:
    welcome_dialog()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    return joblib.load(path)


def extract_urls(text: str):
    found = URL_PATTERN.findall(text)
    if not found:
        return "No URL", 0
    return " ".join(found), len(found)


def validate_input_frame(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    return (not missing and not extra), missing, extra


# ----------------------------------------------------------------------
# Custom CSS (Clean, Professional, No AI Gradients)
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-page: #F8FAFC;
        --card-bg: #FFFFFF;
        --border-subtle: #E2E8F0;
        --border-strong: #CBD5E1;
        --text-main: #0F172A;
        --text-muted: #475569;
        --accent-emerald: #0D7A57;
        --accent-emerald-soft: #ECFDF5;
        --accent-danger: #B91C1C;
        --accent-danger-soft: #FEF2F2;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background-color: var(--bg-page);
        color: var(--text-main);
    }

    #MainMenu, header, footer { visibility: hidden; }
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 720px;
    }

    /* Titles */
    .app-title {
        font-weight: 700;
        font-size: 1.65rem;
        color: var(--text-main);
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        color: var(--text-muted);
        font-size: 0.92rem;
        margin-bottom: 1.6rem;
    }

    /* Main Container Card */
    .input-card {
        background: var(--card-bg);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1.4rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        margin-bottom: 1.2rem;
    }

    /* High Visibility Textarea */
    .stTextArea textarea {
        background-color: #F8FAFC !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: 8px !important;
        color: #0F172A !important;
        font-size: 0.92rem !important;
        line-height: 1.5 !important;
        padding: 0.85rem !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--accent-emerald) !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 0 0 1px var(--accent-emerald) !important;
    }

    .field-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.4rem;
    }

    /* Segmented Controls */
    div[data-testid="stSegmentedControl"] button {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: var(--text-muted) !important;
        border-color: var(--border-subtle) !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
        background-color: #0F172A !important;
        color: #FFFFFF !important;
        border-color: #0F172A !important;
    }

    /* Primary Action Button */
    .stButton button {
        background-color: var(--accent-emerald) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.55rem 1.6rem !important;
        transition: opacity 0.15s ease;
    }
    .stButton button:hover {
        opacity: 0.9;
    }

    /* Result Strip Cards */
    .result-card {
        border-radius: 10px;
        padding: 1.2rem 1.3rem;
        border: 1px solid var(--border-subtle);
        background: #FFFFFF;
        margin-top: 1rem;
    }
    .result-card.phishing {
        border-color: #FCA5A5;
        background-color: var(--accent-danger-soft);
    }
    .result-card.legit {
        border-color: #A7F3D0;
        background-color: var(--accent-emerald-soft);
    }

    .result-title {
        font-weight: 700;
        font-size: 1.25rem;
    }
    .result-title.phishing { color: var(--accent-danger); }
    .result-title.legit { color: var(--accent-emerald); }

    .result-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: var(--text-muted);
        margin-top: 0.4rem;
    }

    .usage-tip {
        font-size: 0.77rem;
        color: var(--text-muted);
        margin-top: 0.55rem;
        line-height: 1.5;
    }

    /* Small Professional Footer */
    .app-footer {
        text-align: center;
        margin-top: 3.5rem;
        padding-top: 1.2rem;
        border-top: 1px solid var(--border-subtle);
        font-size: 0.78rem;
        color: var(--text-muted);
    }
    .app-footer a {
        color: #0F172A;
        font-weight: 600;
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown('<div class="app-title">🛡️ FoxLens Phishing Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Paste an email text to evaluate phishing risk with calibrated ML classifiers.</div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Input Panel
# ----------------------------------------------------------------------
st.markdown('<div class="input-card">', unsafe_allow_html=True)

st.markdown('<div class="field-label">Email Content</div>', unsafe_allow_html=True)
email_text = st.text_area(
    "Email text",
    height=170,
    placeholder="Paste the full email subject and body here...",
    label_visibility="collapsed",
)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

col_attach, col_links = st.columns(2)
with col_attach:
    st.markdown('<div class="field-label">Has Attachment</div>', unsafe_allow_html=True)
    has_attachment_choice = st.segmented_control(
        "Attachment", options=["No", "Yes"], default="No", label_visibility="collapsed"
    )
with col_links:
    st.markdown('<div class="field-label">URL Extraction</div>', unsafe_allow_html=True)
    url_mode = st.segmented_control(
        "Links", options=["Automatic", "Manual"], default="Automatic", label_visibility="collapsed"
    )

manual_url_count = None
manual_urls_text = ""
if url_mode == "Manual":
    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
    m_count, m_urls = st.columns([1, 2])
    with m_count:
        manual_url_count = st.number_input(
            "Count", min_value=0, value=1, step=1, label_visibility="collapsed"
        )
    with m_urls:
        manual_urls_text = st.text_input(
            "Urls", placeholder="Links separated by spaces", label_visibility="collapsed"
        )

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
st.markdown('<div class="field-label">Model Selection</div>', unsafe_allow_html=True)
model_choice = st.segmented_control(
    "Model",
    options=["Lite", "NexusVector-X", "Compare Both"],
    default="Lite",
    label_visibility="collapsed",
)
st.markdown(f'<div class="usage-tip">{USAGE_TIP}</div>', unsafe_allow_html=True)

st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)
analyze_clicked = st.button("Run Security Analysis", use_container_width=False)

st.markdown("</div>", unsafe_allow_html=True)  # end input-card

# ----------------------------------------------------------------------
# Execution & Prediction
# ----------------------------------------------------------------------
if analyze_clicked:
    if not email_text or not email_text.strip():
        st.warning("Please paste an email before running the analysis.")
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
        st.error(f"Input validation error. Expected columns: {REQUIRED_COLUMNS}")
        st.stop()

    def run_single_model(model_key):
        info = MODEL_FILES[model_key]
        clf = load_model(info["path"])
        p = clf.predict_proba(input_df)[:, 1][0]
        pred = int(p >= clf.threshold)
        conf = p if pred == 1 else (1 - p)
        return {
            "key": model_key,
            "family": info["family"],
            "prediction": pred,
            "label": "Phishing Detected" if pred == 1 else "Legitimate Email",
            "confidence": conf,
        }

    with st.spinner("Analyzing email patterns..."):
        if model_choice == "Compare Both":
            res_lite = run_single_model("Lite")
            res_nexus = run_single_model("NexusVector-X")

            col_a, col_b = st.columns(2)
            for col, res in [(col_a, res_lite), (col_b, res_nexus)]:
                status_cls = "phishing" if res["prediction"] == 1 else "legit"
                with col:
                    st.markdown(
                        f"""
                        <div class="result-card {status_cls}">
                            <div class="result-title {status_cls}">{res["label"]}</div>
                            <div class="result-meta">
                                <b>Model:</b> {res["key"]}<br>
                                <b>Confidence:</b> {res["confidence"] * 100:.1f}%<br>
                                <span style="font-size:0.75rem;">({res["family"]})</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            res = run_single_model(model_choice)
            status_cls = "phishing" if res["prediction"] == 1 else "legit"

            st.markdown(
                f"""
                <div class="result-card {status_cls}">
                    <div class="result-title {status_cls}">{res["label"]}</div>
                    <div class="result-meta">
                        Confidence: <b>{res["confidence"] * 100:.1f}%</b> &nbsp;•&nbsp;
                        Model: {res["key"]} ({res["family"]})
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        f'<div style="font-size:0.75rem; color:#64748B; margin-top:0.8rem; line-height:1.4;">ℹ️ {DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------
# Professional Discrete Footer
# ----------------------------------------------------------------------
st.markdown(
    f"""
    <div class="app-footer">
        Designed & Developed by Farouk Ouici &nbsp;•&nbsp; 
        Need a customized AI or security solution? 
        <a href="{FIVERR_PROFILE_URL}" target="_blank">Contact me on Fiverr</a>
    </div>
    """,
    unsafe_allow_html=True,
)
