"""
OmniCourt-AI Third Umpire Intelligence Module.
Uses gemini-3.1-flash-lite on uncropped RGB frames to adjudicate Run Outs, Stumpings, & Bowled dismissals.
Features formal ICC Stadium & Broadcast Voice Transmission synthesis.
"""

import os

# Block GCE metadata server lookup and force standard Developer API
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

import json
from typing import Literal, List, Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

import cv_engine

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class AdjudicationDocket(BaseModel):
    decision: Literal['OUT', 'NOT OUT'] = Field(description="Final DRS decision: 'OUT' or 'NOT OUT'.")
    critical_timestamp_sec: float = Field(description="Timestamp in seconds when bails are dislodged or wicket is broken.")
    bat_grounded_behind_crease: bool = Field(description="True if bat or batter is grounded with physical contact past the popping crease line.")
    governing_mcc_law: str = Field(description="Primary governing MCC Law (e.g., 'MCC Law 38.1 (Run Out)', 'MCC Law 39.1 (Stumped)', 'MCC Law 32.1 (Bowled)').")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0.")
    visual_evidence_summary: str = Field(description="Clear technical explanation detailing stump impact, bails status, crease line, and bat/foot grounding.")
    agent_reasoning_trace: List[str] = Field(description="Step-by-step reasoning trace establishing the verdict.")
    umpire_broadcast_audio_script: str = Field(
        description="Official formal stadium announcement spoken with clarity and authority."
    )

    @property
    def confidence(self) -> float:
        return self.confidence_score

    @property
    def wicket_break_time(self) -> float:
        return self.critical_timestamp_sec

    @property
    def rationale(self) -> str:
        return self.visual_evidence_summary

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        d["confidence"] = self.confidence_score
        d["wicket_break_time"] = self.critical_timestamp_sec
        d["rationale"] = self.visual_evidence_summary
        return d


def clean_key(val: Optional[str]) -> str:
    """Strips quotes, spaces, and newline characters from API keys."""
    if not val:
        return ""
    return str(val).strip().strip('"').strip("'").strip()


def get_effective_api_key(api_key_override: Optional[str] = None) -> str:
    """
    Uniform single source of truth for API key resolution:
    1. Manual UI override if entered.
    2. GEMINI_API_KEY environment variable.
    3. GOOGLE_API_KEY environment variable.
    4. Empty string if none detected.
    """
    cleaned_override = clean_key(api_key_override)
    if cleaned_override:
        return cleaned_override

    env_gemini = clean_key(os.environ.get("GEMINI_API_KEY"))
    if env_gemini:
        return env_gemini

    env_google = clean_key(os.environ.get("GOOGLE_API_KEY"))
    if env_google:
        return env_google

    return ""


def adjudicate_clip(
    video_path: str,
    fps_sample: float = 15.0,
    extracted_frames: Optional[List[Dict[str, Any]]] = None,
    api_key_override: Optional[str] = None,
    evidence_frames: Optional[List[Dict[str, Any]]] = None,
) -> AdjudicationDocket:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    frames = extracted_frames or cv_engine.extract_video_frames(video_path, fps_sample=fps_sample, max_width=720)
    target_frames = evidence_frames if (evidence_frames and len(evidence_frames) > 0) else frames

    active_key = get_effective_api_key(api_key_override)
    if not active_key:
        raise RuntimeError("Gemini API key is missing. Set GEMINI_API_KEY in deployment settings or enter a Live Override.")

    if not GENAI_AVAILABLE:
        raise RuntimeError("The google-genai library is not installed.")

    client = genai.Client(
        api_key=active_key,
        vertexai=False,
        http_options=types.HttpOptions(
            headers={"x-goog-api-key": active_key}
        )
    )

    sample_frames = target_frames
    if len(target_frames) > 5:
        indices = [int(round(i)) for i in np.linspace(0, len(target_frames) - 1, num=5)]
        sample_frames = [target_frames[i] for i in sorted(list(set(indices)))]

    pil_images = []
    timecodes = []
    for f_info in sample_frames:
        f_path = f_info.get("file_path")
        t_sec = f_info.get("timestamp_sec", 0.0)
        timecodes.append(t_sec)
        if f_path and os.path.exists(f_path):
            img_bgr = cv2.imread(f_path)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                pil_images.append(Image.fromarray(img_rgb))

    n = len(pil_images)
    break_est = timecodes[len(timecodes) // 2] if timecodes else 0.0

    prompt = f"""
You are the Official ICC Elite Panel Third Umpire delivering a formal review adjudication.
You are inspecting {n} sequential broadcast frames at timestamps: {timecodes}.

DECISION PROTOCOL:
1. CHECK FOR DIRECT "BOWLED" (MCC Law 32).
2. IDENTIFY THE EXACT POINT OF WICKET BREAK (MCC Law 29).
3. EVALUATE CREASE GROUNDING (MCC Law 38 Run Out / MCC Law 39 Stumped):
   - The line belongs to the umpire.
   - For "NOT OUT": bat tip or batter is physically grounded on turf COMPLETELY PAST popping crease before bails dislodge.
   - Otherwise rule "OUT".

STADIUM BROADCAST ANNOUNCEMENT SCRIPT:
In the field `umpire_broadcast_audio_script`, provide an authoritative, professional verbal statement suitable for live stadium broadcast, following this structure:
"Official DRS Adjudication: Evaluating crease geometry at bail dislodgment. At timestamp {break_est:.2f} seconds, bails are completely broken from the stumps. Visual evidence confirms the bat is [clearly grounded past the popping crease / airborne on the line with no turf contact]. Ruling under [MCC Law]: [OUT / NOT OUT]. Decision confirmed on the big screen."

Return strictly a JSON object conforming to this schema:
{{
  "decision": "OUT" or "NOT OUT",
  "critical_timestamp_sec": {break_est},
  "bat_grounded_behind_crease": true or false,
  "governing_mcc_law": "MCC Law 38.1 (Run Out)" or "MCC Law 39.1 (Stumped)" or "MCC Law 32.1 (Bowled)",
  "confidence_score": 0.98,
  "visual_evidence_summary": "Technical finding detailing bail displacement and bat grounding.",
  "agent_reasoning_trace": [
    "Step 1: Frame sequence ingested.",
    "Step 2: Microsecond bail break confirmed.",
    "Step 3: Ground contact geometry verified."
  ],
  "umpire_broadcast_audio_script": "Official DRS Adjudication: Evaluating crease geometry..."
}}
"""

    contents = pil_images + [prompt]

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=AdjudicationDocket
        )
    )

    data = json.loads(response.text)
    return AdjudicationDocket(**data)