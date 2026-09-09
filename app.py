"""
OmniCourt-AI DRS Broadcast Cockpit.
High-precision third-umpire interface with Gemini 3.1 Flash-Lite video scanning and adjudication.
"""

import os
import hashlib

# Load local environment if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st
from PIL import Image

import cv_engine
from umpire_agent import adjudicate_clip, AdjudicationDocket, get_effective_api_key, clean_key

st.set_page_config(page_title="OmniCourt-AI | DRS Studio", page_icon="🏏", layout="wide")

# Custom Broadcast Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=Space+Grotesk:wght@600;700&family=JetBrains+Mono:wght@500;700&display=swap');
    
    .stApp {
        background: radial-gradient(circle at 50% 0%, #0D1527 0%, #070B14 60%, #04070D 100%);
        color: #E2E8F0;
        font-family: 'Outfit', sans-serif;
    }
    .studio-navbar {
        background: linear-gradient(90deg, rgba(13, 21, 39, 0.95) 0%, rgba(20, 30, 55, 0.85) 50%, rgba(10, 16, 30, 0.95) 100%);
        border: 1px solid rgba(0, 240, 255, 0.25);
        border-radius: 14px;
        padding: 16px 24px;
        margin-bottom: 22px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .studio-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 24px;
        font-weight: 800;
        letter-spacing: 1px;
        color: #FFFFFF;
        margin: 0;
    }
    .studio-title span { color: #00F0FF; }
    .studio-card-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 15px;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        margin-bottom: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 6px;
    }
    .banner-out {
        background: linear-gradient(135deg, rgba(225, 29, 72, 0.25) 0%, rgba(136, 19, 55, 0.4) 100%);
        border: 2px solid #FF1744;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        margin-top: 15px;
    }
    .banner-notout {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.25) 0%, rgba(6, 78, 59, 0.4) 100%);
        border: 2px solid #00E676;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        margin-top: 15px;
    }
    .verdict-text {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 52px;
        font-weight: 900;
        letter-spacing: 4px;
        line-height: 1;
        margin: 0;
    }
    .verdict-out-color { color: #FF1744; }
    .verdict-notout-color { color: #00E676; }
</style>
""", unsafe_allow_html=True)


def get_key_fingerprint(key_str: str) -> str:
    """Returns a non-reversible SHA-256 fingerprint for safe diagnostics."""
    if not key_str:
        return "None"
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:12]


# Session State Initialization - Live Override widget starts EMPTY as an optional field
if "gemini_key_widget" not in st.session_state:
    st.session_state["gemini_key_widget"] = ""
if "current_video_path" not in st.session_state:
    st.session_state.current_video_path = None
if "extracted_frames" not in st.session_state:
    st.session_state.extracted_frames = []
if "selected_frame_idx" not in st.session_state:
    st.session_state.selected_frame_idx = 0
if "locked_impact_frame_idx" not in st.session_state:
    st.session_state.locked_impact_frame_idx = 0
if "adjudication_result" not in st.session_state:
    st.session_state.adjudication_result = None
if "adjudication_error" not in st.session_state:
    st.session_state.adjudication_error = None
if "auto_event_data" not in st.session_state:
    st.session_state.auto_event_data = None

# Navbar
st.markdown("""
<div class="studio-navbar">
    <div>
        <h1 class="studio-title">🏏 <span>OmniCourt-AI</span> DRS Studio</h1>
        <div style="font-size: 12px; color: #94A3B8;">Third Umpire • Gemini 3.1 Flash-Lite Engine</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Key Resolution
override_key = clean_key(st.session_state.get("gemini_key_widget", ""))
env_gemini_key = clean_key(os.environ.get("GEMINI_API_KEY"))
env_google_key = clean_key(os.environ.get("GOOGLE_API_KEY"))
active_key = override_key or env_gemini_key or env_google_key

# Sidebar
with st.sidebar:
    st.subheader("DRS Configuration")

    st.text_input(
        "Gemini API Key (Live Override)",
        key="gemini_key_widget",
        type="password",
        help="Optional: Enter a key here to override the deployment environment secret."
    )

    # Safe Diagnostic Panel
    with st.expander("🔍 Authentication Diagnostics", expanded=True):
        if override_key:
            st.success("State: **LIVE OVERRIDE PROVIDED**")
            st.caption(f"Length: `{len(override_key)}` chars | SHA-256: `{get_key_fingerprint(override_key)}`")
        elif env_gemini_key:
            st.info("State: **DEFAULT ENVIRONMENT KEY AVAILABLE**")
            st.caption(f"Source: `GEMINI_API_KEY` | Length: `{len(env_gemini_key)}` chars | SHA-256: `{get_key_fingerprint(env_gemini_key)}`")
        elif env_google_key:
            st.info("State: **DEFAULT ENVIRONMENT KEY AVAILABLE**")
            st.caption(f"Source: `GOOGLE_API_KEY` | Length: `{len(env_google_key)}` chars | SHA-256: `{get_key_fingerprint(env_google_key)}`")
        else:
            st.error("State: **KEY MISSING**")
            st.caption("Neither Live Override nor container GEMINI_API_KEY detected.")

        st.caption(f"Vertex AI Mode: `{os.environ.get('GOOGLE_GENAI_USE_VERTEXAI', 'False')}`")

    videos_dir = os.path.join("assets", "videos")
    os.makedirs(videos_dir, exist_ok=True)
    existing_videos = [f for f in sorted(os.listdir(videos_dir)) if f.endswith((".mp4", ".mov", ".avi"))]

    video_choice = st.selectbox("Select Match Incident", ["Upload Custom Video..."] + existing_videos)
    uploaded_file = None
    if video_choice == "Upload Custom Video...":
        uploaded_file = st.file_uploader("Upload Incident Clip", type=["mp4", "mov", "avi"])

    active_video_path = None
    if video_choice == "Upload Custom Video..." and uploaded_file is not None:
        save_path = os.path.join(videos_dir, f"uploaded_{uploaded_file.name}")
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        active_video_path = save_path
    elif video_choice != "Upload Custom Video...":
        active_video_path = os.path.join(videos_dir, video_choice)

# Automatic pipeline on video selection
if active_video_path and active_video_path != st.session_state.current_video_path:
    st.session_state.current_video_path = active_video_path
    st.session_state.adjudication_result = None
    st.session_state.adjudication_error = None

    with st.spinner("Extracting synchronized video frames..."):
        base_name = os.path.splitext(os.path.basename(active_video_path))[0]
        st.session_state.extracted_frames = cv_engine.extract_video_frames(
            active_video_path,
            output_dir=os.path.join("assets", "frames", base_name),
            fps_sample=15.0,
            max_width=720
        )

    with st.spinner("Scanning video with Gemini 3.1 Flash-Lite to pinpoint wicket break..."):
        event_data = cv_engine.find_event_timestamp_with_ai(
            active_video_path,
            api_key=active_key,
            frames=st.session_state.extracted_frames
        )
        st.session_state.auto_event_data = event_data
        idx = event_data.get("impact_frame_index", len(st.session_state.extracted_frames) // 2)
        st.session_state.locked_impact_frame_idx = idx
        st.session_state.selected_frame_idx = idx

frames = st.session_state.extracted_frames

col_left, col_right = st.columns([1.15, 1.0], gap="large")

with col_left:
    st.markdown('<div class="studio-card-header">🎥 Incident Feed & Frame Scrubber</div>', unsafe_allow_html=True)
    if active_video_path:
        st.video(active_video_path, format="video/mp4")

    if frames:
        num_frames = len(frames)
        cur_idx = min(st.session_state.selected_frame_idx, num_frames - 1)

        slider_val = st.slider(
            "Timeline Scrubber",
            0,
            num_frames - 1,
            cur_idx,
            format="Frame %d",
            label_visibility="collapsed"
        )
        if slider_val != st.session_state.selected_frame_idx:
            st.session_state.selected_frame_idx = slider_val
            st.rerun()

        n1, n2, n3, n4 = st.columns(4)
        if n1.button("⏮ -5 Frames", use_container_width=True):
            st.session_state.selected_frame_idx = max(0, cur_idx - 5)
            st.rerun()
        if n2.button("◀ -1 Frame", use_container_width=True):
            st.session_state.selected_frame_idx = max(0, cur_idx - 1)
            st.rerun()
        if n3.button("+1 Frame ▶", use_container_width=True):
            st.session_state.selected_frame_idx = min(num_frames - 1, cur_idx + 1)
            st.rerun()
        if n4.button("+5 Frames ⏭", use_container_width=True):
            st.session_state.selected_frame_idx = min(num_frames - 1, cur_idx + 5)
            st.rerun()

        if st.button("📌 Lock Current Frame as Impact Point", use_container_width=True):
            st.session_state.locked_impact_frame_idx = st.session_state.selected_frame_idx
            st.rerun()

        frame_data = frames[st.session_state.selected_frame_idx]
        if os.path.exists(frame_data["file_path"]):
            st.image(Image.open(frame_data["file_path"]), use_container_width=True)
            st.caption(
                f"Timecode: T={frame_data['timestamp_sec']:.2f}s | Frame #{st.session_state.selected_frame_idx + 1}"
            )

with col_right:
    st.markdown('<div class="studio-card-header">📸 Visual Evidence Strip (Target Payload)</div>', unsafe_allow_html=True)

    evidence_frames = []
    if frames:
        evidence_frames = cv_engine.get_evidence_frames(
            frames,
            center_idx=st.session_state.locked_impact_frame_idx,
            frames_before=2,
            frames_after=2,
            step_interval=1,
            max_total=5
        )

    if evidence_frames:
        ev_cols = st.columns(len(evidence_frames))
        for i, ef in enumerate(evidence_frames):
            with ev_cols[i]:
                is_impact = (ef["saved_index"] == st.session_state.locked_impact_frame_idx)
                lbl = f"T={ef['timestamp_sec']:.2f}s" + (" 🎯 IMPACT" if is_impact else "")
                st.caption(lbl)
                st.image(Image.open(ef["file_path"]), use_container_width=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    if st.button("🔴 Send to Third Umpire (Adjudicate)", type="primary", use_container_width=True):
        if not active_key:
            st.session_state.adjudication_error = "Gemini API key is missing. Set GEMINI_API_KEY in the environment or enter a Live Override."
            st.session_state.adjudication_result = None
            st.rerun()

        with st.spinner("Gemini 3.1 Flash-Lite is evaluating crease geometry and bails..."):
            try:
                docket = adjudicate_clip(
                    active_video_path,
                    extracted_frames=frames,
                    api_key_override=active_key,
                    evidence_frames=evidence_frames
                )
                st.session_state.adjudication_result = docket
                st.session_state.adjudication_error = None
            except Exception as e:
                st.session_state.adjudication_error = str(e)
                st.session_state.adjudication_result = None
            st.rerun()

    if st.session_state.get("adjudication_error"):
        st.error(f"Adjudication Error: {st.session_state.adjudication_error}")

    docket: AdjudicationDocket = st.session_state.adjudication_result
    if docket:
        banner_cls = "banner-out" if docket.decision == "OUT" else "banner-notout"
        txt_cls = "verdict-out-color" if docket.decision == "OUT" else "verdict-notout-color"
        st.markdown(f"""
        <div class="{banner_cls}">
            <div class="verdict-text {txt_cls}">{docket.decision}</div>
            <div style="font-size: 13px; font-weight: 700; color: #CBD5E1; margin-top: 6px;">
                {docket.governing_mcc_law} • Confidence: {docket.confidence_score * 100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📋 Visual Evidence & Physical Rationale")
        st.write(docket.visual_evidence_summary)

        with st.expander("🤖 Multi-Agent Deliberation Trace", expanded=True):
            for step in docket.agent_reasoning_trace:
                st.markdown(f"- {step}")