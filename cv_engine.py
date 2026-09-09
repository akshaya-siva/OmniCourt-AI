"""
OmniCourt-AI Computer Vision Engine.
Handles frame extraction, temporal scanning with Gemini 3.1 Flash-Lite, and visual evidence slicing.
"""

import os

# Enforce Gemini Developer API and disable Vertex AI auto-detection
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

import cv2
import json
import numpy as np
from typing import List, Dict, Any, Optional

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

DEFAULT_GEMINI_KEY = ""


def clean_key(val: Optional[str]) -> str:
    """Strips quotes, spaces, and newline characters that break API keys."""
    if not val:
        return ""
    return str(val).strip().strip('"').strip("'").strip()


def extract_video_frames(
    video_path: str,
    output_dir: Optional[str] = None,
    fps_sample: float = 15.0,
    max_width: int = 720
) -> List[Dict[str, Any]]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sample_interval = max(1, int(round(orig_fps / fps_sample)))

    frame_list = []
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            h, w = frame.shape[:2]
            if w > max_width:
                scaling = max_width / float(w)
                new_w, new_h = max_width, int(h * scaling)
                frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                frame_resized = frame

            timestamp_sec = frame_idx / orig_fps
            file_path = None
            if output_dir:
                file_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(file_path, frame_resized)

            frame_list.append({
                "saved_index": saved_count,
                "raw_frame_index": frame_idx,
                "timestamp_sec": float(timestamp_sec),
                "file_path": file_path
            })
            saved_count += 1

        frame_idx += 1

    cap.release()
    return frame_list


def find_event_timestamp_with_ai(
    video_path: str,
    api_key: Optional[str] = None,
    frames: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    num_frames = len(frames) if frames else 10
    default_idx = num_frames // 2
    default_time = frames[default_idx]["timestamp_sec"] if frames else 1.5

    fallback_data = {
        "impact_timestamp_sec": default_time,
        "impact_frame_index": default_idx,
        "notes": "Default impact estimation"
    }

    cleaned = clean_key(api_key)
    effective_key = cleaned or clean_key(os.environ.get("GEMINI_API_KEY")) or DEFAULT_GEMINI_KEY

    if not effective_key or not GENAI_AVAILABLE:
        return fallback_data

    try:
        # Enforce explicit header and vertexai=False
        client = genai.Client(
            api_key=effective_key,
            vertexai=False,
            http_options=types.HttpOptions(headers={"x-goog-api-key": effective_key})
        )
        uploaded_file = client.files.upload(file=video_path)

        prompt = """
Analyze this cricket incident clip.
Identify the exact time in seconds when the wicket is broken.
Respond strictly in JSON:
{
  "impact_timestamp_sec": 1.50
}
"""
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=[uploaded_file, prompt],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json"
            )
        )

        data = json.loads(response.text)
        impact_t = float(data.get("impact_timestamp_sec", default_time))

        best_idx = default_idx
        if frames:
            diffs = [abs(f["timestamp_sec"] - impact_t) for f in frames]
            best_idx = int(np.argmin(diffs))

        return {
            "impact_timestamp_sec": impact_t,
            "impact_frame_index": best_idx,
            "notes": "AI-detected impact frame"
        }
    except Exception:
        # Silently fall back to geometric midpoint if scanning fails
        return fallback_data


def get_evidence_frames(
    frames: List[Dict[str, Any]],
    center_idx: int,
    frames_before: int = 2,
    frames_after: int = 2,
    step_interval: int = 1,
    max_total: int = 5
) -> List[Dict[str, Any]]:
    if not frames:
        return []

    center_idx = max(0, min(center_idx, len(frames) - 1))
    start_idx = max(0, center_idx - (frames_before * step_interval))
    end_idx = min(len(frames), center_idx + (frames_after * step_interval) + 1)

    selected = [frames[i] for i in range(start_idx, end_idx, step_interval)]

    if len(selected) > max_total:
        indices = [int(round(x)) for x in np.linspace(0, len(selected) - 1, max_total)]
        selected = [selected[i] for i in sorted(list(set(indices)))]

    return selected