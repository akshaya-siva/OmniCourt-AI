"""
OmniCourt-AI Computer Vision Engine.
Provides frame extraction utilities and Gemini 3.1 Flash-Lite video scanning via Google GenAI Files API.
"""

import os
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

import cv2
import numpy as np
import time
import json
from typing import List, Dict, Any, Optional

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


def extract_video_frames(
    video_path: str,
    output_dir: str = "assets/frames",
    fps_sample: Optional[float] = None,
    frame_interval: Optional[int] = None,
    max_frames: Optional[int] = None,
    max_width: Optional[int] = 720,
) -> List[Dict[str, Any]]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps_sample is not None and fps_sample > 0:
        step = max(1, int(round(video_fps / fps_sample)))
    elif frame_interval is not None and frame_interval > 0:
        step = frame_interval
    else:
        step = max(1, int(round(video_fps)))

    extracted_frames: List[Dict[str, Any]] = []
    current_frame = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame % step == 0:
            h, w = frame.shape[:2]
            if max_width and w > max_width:
                scale = max_width / float(w)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (max_width, new_h), interpolation=cv2.INTER_AREA)
                h, w = frame.shape[:2]

            timestamp_sec = round(current_frame / video_fps, 3)
            filename = f"frame_{saved_count:04d}_t{timestamp_sec:.2f}s.jpg"
            frame_path = os.path.join(output_dir, filename)

            cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

            extracted_frames.append({
                "frame_index": current_frame,
                "saved_index": saved_count,
                "timestamp_sec": timestamp_sec,
                "file_path": frame_path,
                "width": w,
                "height": h,
            })

            saved_count += 1
            if max_frames and saved_count >= max_frames:
                break

        current_frame += 1

    cap.release()
    return extracted_frames


def find_event_timestamp_with_ai(
    video_path: str,
    api_key: Optional[str] = None,
    frames: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Direct implementation using Gemini 3.1 Flash-Lite and the Files API.
    Identifies the exact second of bail dislodgment across the full match video.
    """
    effective_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
    if not effective_key or not GENAI_AVAILABLE:
        return {"impact_timestamp_sec": 0.0, "impact_frame_index": 0, "source": "Manual Standby"}

    try:
        client = genai.Client(api_key=effective_key)
        video_file = client.files.upload(file=video_path)

        while video_file.state.name == "PROCESSING":
            time.sleep(1.5)
            video_file = client.files.get(name=video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError("Gemini File API failed to process video.")

        prompt = """
        You are an expert cricket DRS video analyst. Watch this clip and detect the exact moment of a potential dismissal.

        TASK: Locate the EXACT decimal timestamp in seconds when the WICKET IS BROKEN.
        
        LOOK SPECIFICALLY FOR:
          - The exact frame the bails first separate from the stumps, fly into the air, or the LED stumps flash red.
          - Could be caused by:
            1. Bowler delivery hitting the stumps directly (Bowled).
            2. Fielder throw hitting the stumps directly (Run Out).
            3. Fielder/Wicketkeeper breaking the stumps with ball in hand (Run Out / Stumping).

        STRICT EXCLUSIONS:
          - Do NOT report when the bowler releases the ball.
          - Do NOT report when the batter hits or misses the ball.
          - Do NOT report player celebrations or umpire signals.
          - Report ONLY the physical moment of BAIL DISLODGEMENT / WICKET BREAK.

        Respond ONLY with this JSON:
        {"timestamp_seconds": 2.45}
        """

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=[video_file, prompt],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json"
            )
        )

        try:
            client.files.delete(name=video_file.name)
        except Exception:
            pass

        data = json.loads(response.text)
        impact_t = float(data.get("timestamp_seconds") or 0.0)

        closest_idx = 0
        if frames and len(frames) > 0:
            closest_idx = min(range(len(frames)), key=lambda i: abs(frames[i]["timestamp_sec"] - impact_t))

        return {
            "impact_timestamp_sec": impact_t,
            "impact_frame_index": closest_idx,
            "source": "Gemini 3.1 Flash-Lite Video API",
            "description": f"Wicket-break detected at T={impact_t:.2f}s"
        }
    except Exception as e:
        print(f"[OmniCourt] AI event timestamp detection error: {e}")
        return {"impact_timestamp_sec": 0.0, "impact_frame_index": 0, "source": "Manual Standby"}


def get_evidence_frames(
    extracted_frames: List[Dict[str, Any]],
    center_idx: int,
    frames_before: int = 2,
    frames_after: int = 2,
    step_interval: int = 1,
    max_total: int = 5,
) -> List[Dict[str, Any]]:
    if not extracted_frames:
        return []

    num_available = len(extracted_frames)
    center_idx = max(0, min(num_available - 1, center_idx))

    target_indices = []
    for b in range(frames_before, 0, -1):
        idx = center_idx - (b * step_interval)
        if idx >= 0:
            target_indices.append(idx)

    target_indices.append(center_idx)

    for a in range(1, frames_after + 1):
        idx = center_idx + (a * step_interval)
        if idx < num_available:
            target_indices.append(idx)

    target_indices = sorted(list(set(target_indices)))

    while len(target_indices) > max_total:
        dist_first = abs(target_indices[0] - center_idx)
        dist_last = abs(target_indices[-1] - center_idx)
        if dist_first >= dist_last:
            target_indices.pop(0)
        else:
            target_indices.pop(-1)

    return [extracted_frames[i] for i in target_indices]