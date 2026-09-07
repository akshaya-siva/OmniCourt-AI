"""
OmniCourt-AI Third Umpire Intelligence Module.
Uses gemini-3.1-flash-lite on uncropped RGB frames to adjudicate Run Outs, Stumpings, & Bowled dismissals.
"""

import os
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

import json
from typing import Literal, List, Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from dotenv import load_dotenv

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


def get_effective_api_key(api_key_override: Optional[str] = None) -> Optional[str]:
    if api_key_override and api_key_override.strip():
        return api_key_override.strip()
    load_dotenv(override=True)
    key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    return key if key else None


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

    api_key = get_effective_api_key(api_key_override)
    if not api_key or not GENAI_AVAILABLE:
        raise RuntimeError("GEMINI_API_KEY is not configured or google-genai is missing.")

    client = genai.Client(api_key=api_key)

    # Convert full, uncropped frames into PIL images (RGB)
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
You are an expert ICC/MCC Elite Panel Third Umpire conducting a DRS video review.
You are inspecting {n} sequential chronological broadcast frames at timestamps: {timecodes}.

DECISION PROTOCOL (Follow strictly in sequence):

1. CHECK FOR DIRECT "BOWLED" (MCC Law 32):
   - Did the bowler's delivered ball strike the wicket directly and dislodge the bails?
   - If YES -> The batter is BOWLED. Crease grounding is irrelevant.
   - Verdict: "OUT", Law: "MCC Law 32.1 (Bowled)", bat_grounded_behind_crease: false.

2. IDENTIFY THE MOMENT THE WICKET IS BROKEN (MCC Law 29):
   - Pinpoint the exact frame where:
     * The bails are first lifted or separated from the stumps, OR
     * The stumps light up (LED Zing bails), OR
     * A stump is struck and displaced by the ball or hand with the ball.

3. CHECK BAT / FOOT GROUNDING (MCC Law 38 Run Out / MCC Law 39 Stumped):
   - Evaluate the batsman's bat tip or foot relative to the white POPPING CREASE line at or prior to the wicket-break frame.
   - OFFICIAL CRICKET RULE: "The line belongs to the umpire."
     * Touching ON the line is NOT safe.
     * Being airborne over the line is NOT safe.
   - For "NOT OUT":
     * There MUST be clear visual evidence that the bat tip or a part of the batter's person is physically grounded ON THE TURF COMPLETELY PAST the popping crease (towards the wicket-keeper/stumps side) before the bails dislodge.
   - For "OUT":
     * The bat/foot is short of the popping crease.
     * The bat/foot is touching the white paint of the crease line, but not grounded beyond it.
     * The bat is sliding above the turf (airborne) as bails break.
     * The foot is raised or sliding in the air (Stumping).
     * If there is doubt or no grounded contact beyond the line -> Rule OUT.

DO NOT default to NOT OUT. If the bat or foot has not visibly made contact with the grass beyond the popping crease line before the bails break, the ruling MUST be OUT.

Return strictly a JSON object conforming to this schema:
{{
  "decision": "OUT" or "NOT OUT",
  "critical_timestamp_sec": {break_est},
  "bat_grounded_behind_crease": true or false,
  "governing_mcc_law": "MCC Law 38.1 (Run Out)" or "MCC Law 39.1 (Stumped)" or "MCC Law 32.1 (Bowled)",
  "confidence_score": 0.95,
  "visual_evidence_summary": "Detailed technical finding: state of bails/stumps, exact position of bat/foot relative to the popping crease, and ground contact.",
  "agent_reasoning_trace": [
    "Step 1: Examined delivery and verified wicket impact.",
    "Step 2: Identified exact frame where bails separate/illuminate.",
    "Step 3: Inspected bat/foot position against the popping crease line."
  ]
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