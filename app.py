"""
OmniCourt-AI DRS Broadcast Cockpit.
ICC Men's T20 World Cup Elite Panel Third Umpire Studio with Gemini 3.1 Flash-Lite Adjudication.
"""

import os
import hashlib
import json

os.environ["NO_GCE_CHECK"] = "True"
os.environ["GCE_METADATA_HOST"] = "none"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

import cv_engine
from umpire_agent import adjudicate_clip, AdjudicationDocket, get_effective_api_key, clean_key

st.set_page_config(
    page_title="OmniCourt-AI | ICC T20 World Cup DRS Studio",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Star Sports / ICC Broadcast Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
    
    .stApp {
        background: radial-gradient(circle at 50% 0%, #0c1731 0%, #060a16 55%, #02040a 100%);
        color: #E2E8F0;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Star Sports Broadcast Header */
    .broadcast-navbar {
        background: linear-gradient(135deg, rgba(12, 23, 49, 0.95) 0%, rgba(20, 36, 75, 0.9) 50%, rgba(8, 14, 30, 0.95) 100%);
        border: 1px solid rgba(0, 240, 255, 0.35);
        border-radius: 16px;
        padding: 18px 28px;
        margin-bottom: 22px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    .broadcast-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 26px;
        font-weight: 900;
        letter-spacing: 1.5px;
        color: #FFFFFF;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0, 240, 255, 0.4);
    }
    .broadcast-title span { color: #00F0FF; }
    .broadcast-badge {
        background: linear-gradient(90deg, #FF1744 0%, #D50000 100%);
        color: #FFFFFF;
        font-weight: 800;
        font-size: 11px;
        letter-spacing: 1.5px;
        padding: 4px 10px;
        border-radius: 6px;
        text-transform: uppercase;
        display: inline-flex;
        align-items: center;
        margin-right: 12px;
        box-shadow: 0 0 12px rgba(255, 23, 68, 0.6);
    }
    .live-pulse {
        width: 8px;
        height: 8px;
        background-color: #FFFFFF;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
        animation: pulse-dot 1.2s infinite ease-in-out;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.3; transform: scale(0.7); }
    }
    
    /* Section Cards */
    .studio-card-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 14px;
        font-weight: 800;
        color: #00F0FF;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 12px;
        border-bottom: 1px solid rgba(0, 240, 255, 0.15);
        padding-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Big Screen TV Decision Banners */
    .banner-out {
        background: linear-gradient(135deg, rgba(225, 29, 72, 0.3) 0%, rgba(136, 19, 55, 0.55) 100%);
        border: 2px solid #FF1744;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-top: 15px;
        box-shadow: 0 0 35px rgba(255, 23, 68, 0.4);
    }
    .banner-notout {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.3) 0%, rgba(6, 78, 59, 0.55) 100%);
        border: 2px solid #00E676;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-top: 15px;
        box-shadow: 0 0 35px rgba(0, 230, 118, 0.4);
    }
    .verdict-text {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 60px;
        font-weight: 900;
        letter-spacing: 6px;
        line-height: 1;
        margin: 0;
    }
    .verdict-out-color { color: #FF1744; text-shadow: 0 0 25px rgba(255, 23, 68, 0.8); }
    .verdict-notout-color { color: #00E676; text-shadow: 0 0 25px rgba(0, 230, 118, 0.8); }
    
    /* Audio Transmission Radio Box */
    .radio-box {
        background: rgba(10, 20, 40, 0.7);
        border: 1px solid rgba(0, 240, 255, 0.3);
        border-left: 4px solid #00F0FF;
        border-radius: 10px;
        padding: 14px 18px;
        margin-top: 16px;
        margin-bottom: 12px;
        font-size: 14px;
        line-height: 1.6;
        color: #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)


def get_key_fingerprint(key_str: str) -> str:
    if not key_str:
        return "None"
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:12]


# Web Speech Audio Synthesizer Component
def play_broadcast_audio_script(text_to_speak: str):
    clean_text = json.dumps(text_to_speak)
    html_code = f"""
    <div style="background: rgba(0, 240, 255, 0.08); border: 1px solid rgba(0, 240, 255, 0.3); border-radius: 12px; padding: 14px 18px; margin-top: 10px; display: flex; align-items: center; justify-content: space-between;">
        <div>
            <div style="font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 14px; color: #00F0FF; letter-spacing: 0.8px;">
                🎙️ ICC THIRD UMPIRE LIVE RADIO STREAM
            </div>
            <div style="font-size: 11px; color: #94A3B8; margin-top: 2px;">
                Direct Elite Panel comms to TV director & big screen
            </div>
        </div>
        <button id="speak-btn" onclick="speakUmpireAudio()" style="background: linear-gradient(135deg, #00F0FF 0%, #0088FF 100%); color: #04070D; border: none; font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 13px; padding: 9px 18px; border-radius: 8px; cursor: pointer; display: flex; align-items: center; gap: 6px; box-shadow: 0 4px 12px rgba(0, 240, 255, 0.35);">
            ▶ PLAY AUDIO TRANSMISSION
        </button>
    </div>

    <script>
    function speakUmpireAudio() {{
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            const text = {clean_text};
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 0.95;
            utterance.pitch = 1.0;
            
            const voices = window.speechSynthesis.getVoices();
            const englishVoice = voices.find(v => v.lang.includes('en') && (v.name.includes('UK') || v.name.includes('British') || v.name.includes('Natural') || v.name.includes('Google')));
            if (englishVoice) {{
                utterance.voice = englishVoice;
            }}
            
            const btn = document.getElementById('speak-btn');
            btn.innerHTML = '🔊 TRANSMITTING...';
            btn.style.background = '#FF1744';
            btn.style.color = '#FFFFFF';
            
            utterance.onend = function() {{
                btn.innerHTML = '🔄 REPLAY TRANSMISSION';
                btn.style.background = 'linear-gradient(135deg, #00F0FF 0%, #0088FF 100%)';
                btn.style.color = '#04070D';
            }};
            
            window.speechSynthesis.speak(utterance);
        }} else {{
            alert('Web Speech Synthesis is not supported in this browser.');
        }}
    }}
    // Trigger automatically on render
    window.addEventListener('load', () => {{
        setTimeout(speakUmpireAudio, 400);
    }});
    </script>
    """
    components.html(html_code, height=90)


# State Initializations
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

# Star Sports Broadcast Top Bar
st.markdown("""
<div class="broadcast-navbar">
    <div style="display: flex; align-items: center;">
        <span class="broadcast-badge"><span class="live-pulse"></span>LIVE DRS</span>
        <div>
            <h1 class="broadcast-title">🏏 <span>OMNICOURT-AI</span> DRS STUDIO</h1>
            <div style="font-size: 11px; font-weight: 600; color: #94A3B8; letter-spacing: 1px;">
                ICC MEN'S T20 WORLD CUP • ELITE PANEL REVIEW SYSTEM • GEMINI 3.1 FLASH-LITE ENGINE
            </div>
        </div>
    </div>
    <div style="display: flex; gap: 10px; font-family: 'Space Grotesk', sans-serif; font-size: 12px; font-weight: 700;">
        <div style="background: rgba(255, 255, 255, 0.06); border: 1px solid rgba(255, 255, 255, 0.12); padding: 8px 14px; border-radius: 8px;">
            ⚡ SUB-FRAME SYNC
        </div>
        <div style="background: rgba(0, 240, 255, 0.12); border: 1px solid rgba(0, 240, 255, 0.35); padding: 8px 14px; border-radius: 8px; color: #00F0FF;">
            📡 HAWK-EYE AI ACTIVE
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Key Resolution
override_key = clean_key(st.session_state.get("gemini_key_widget", ""))
env_gemini_key = clean_key(os.environ.get("GEMINI_API_KEY"))
env_google_key = clean_key(os.environ.get("GOOGLE_API_KEY"))
active_key = override_key or env_gemini_key or env_google_key

# Sidebar Controls
with st.sidebar:
    st.markdown("### ⚙️ DRS Cockpit Settings")
    st.text_input(
        "Gemini API Key (Live Override)",
        key="gemini_key_widget",
        type="password",
        help="Optional: Enter a key here to override the deployment environment secret."
    )

    with st.expander("🔍 Match Comms & Diagnostics", expanded=True):
        if override_key:
            st.success("State: **LIVE OVERRIDE ACTIVE**")
            st.caption(f"Length: `{len(override_key)}` chars | SHA-256: `{get_key_fingerprint(override_key)}`")
        elif env_gemini_key or env_google_key:
            st.info("State: **BROADCAST KEY READY**")
            src = "GEMINI_API_KEY" if env_gemini_key else "GOOGLE_API_KEY"
            st.caption(f"Source: `{src}` | Length: `{len(active_key)}` chars | SHA-256: `{get_key_fingerprint(active_key)}`")
        else:
            st.error("State: **NO KEY DETECTED**")
            st.caption("Please configure `GEMINI_API_KEY` in Cloud Run.")

        st.caption("Model: `gemini-3.1-flash-lite` | Vertex AI: `Disabled`")

    videos_dir = os.path.join("assets", "videos")
    os.makedirs(videos_dir, exist_ok=True)
    existing_videos = [f for f in sorted(os.listdir(videos_dir)) if f.endswith((".mp4", ".mov", ".avi"))]

    video_choice = st.selectbox("Select Broadcast Match Incident", ["Upload Custom Video..."] + existing_videos)
    uploaded_file = None
    if video_choice == "Upload Custom Video...":
        uploaded_file = st.file_uploader("Upload Broadcast Feed Clip", type=["mp4", "mov", "avi"])

    active_video_path = None
    if video_choice == "Upload Custom Video..." and uploaded_file is not None:
        save_path = os.path.join(videos_dir, f"uploaded_{uploaded_file.name}")
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        active_video_path = save_path
    elif video_choice != "Upload Custom Video...":
        active_video_path = os.path.join(videos_dir, video_choice)

# Video Ingestion Pipeline
if active_video_path and active_video_path != st.session_state.current_video_path:
    st.session_state.current_video_path = active_video_path
    st.session_state.adjudication_result = None
    st.session_state.adjudication_error = None

    with st.spinner("Extracting synchronized high-definition video frames..."):
        base_name = os.path.splitext(os.path.basename(active_video_path))[0]
        st.session_state.extracted_frames = cv_engine.extract_video_frames(
            active_video_path,
            output_dir=os.path.join("assets", "frames", base_name),
            fps_sample=15.0,
            max_width=720
        )

    with st.spinner("Scanning footage with Gemini 3.1 Flash-Lite to pinpoint wicket break..."):
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

col_left, col_right = st.columns([1.12, 1.0], gap="large")

with col_left:
    st.markdown('<div class="studio-card-header">📹 Broadcast Incident Feed & Sub-Frame Scrubber</div>', unsafe_allow_html=True)
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

        if st.button("🎯 Lock Current Frame as Bail Dislodgment Point", use_container_width=True):
            st.session_state.locked_impact_frame_idx = st.session_state.selected_frame_idx
            st.rerun()

        frame_data = frames[st.session_state.selected_frame_idx]
        if os.path.exists(frame_data["file_path"]):
            st.image(Image.open(frame_data["file_path"]), use_container_width=True)
            st.caption(
                f"Timecode: T={frame_data['timestamp_sec']:.2f}s | Broadcast Frame #{st.session_state.selected_frame_idx + 1}"
            )

with col_right:
    st.markdown('<div class="studio-card-header">🔍 Visual Evidence Strip (Payload to Gemini 3.1)</div>', unsafe_allow_html=True)

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

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    if st.button("🔴 Send to Third Umpire (Adjudicate Incident)", type="primary", use_container_width=True):
        if not active_key:
            st.session_state.adjudication_error = "Gemini API key is missing. Set GEMINI_API_KEY in Cloud Run or enter a Live Override."
            st.session_state.adjudication_result = None
            st.rerun()

        with st.spinner("Third Umpire is verifying crease line geometry, bat bounce, and bails..."):
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
        # Star Sports Stadium Decision Banner
        banner_cls = "banner-out" if docket.decision == "OUT" else "banner-notout"
        txt_cls = "verdict-out-color" if docket.decision == "OUT" else "verdict-notout-color"
        st.markdown(f"""
        <div class="{banner_cls}">
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 14px; font-weight: 800; color: #CBD5E1; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 4px;">
                ICC T20 WORLD CUP OFFICIAL VERDICT
            </div>
            <div class="verdict-text {txt_cls}">{docket.decision}</div>
            <div style="font-size: 14px; font-weight: 700; color: #FFFFFF; margin-top: 8px;">
                {docket.governing_mcc_law} • Confidence: {docket.confidence_score * 100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Automated ICC Radio Voice Synthesizer
        play_broadcast_audio_script(docket.umpire_broadcast_audio_script)

        # Official Transmission Script Box
        st.markdown(f"""
        <div class="radio-box">
            <strong style="color: #00F0FF;">📻 Transmitted Radio Dialogue:</strong><br>
            <em>"{docket.umpire_broadcast_audio_script}"</em>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📋 Crease Geometry & Physical Rationale")
        st.write(docket.visual_evidence_summary)

        with st.expander("🤖 Multi-Agent Deliberation & Protocol Trace", expanded=True):
            for step in docket.agent_reasoning_trace:
                st.markdown(f"- {step}")