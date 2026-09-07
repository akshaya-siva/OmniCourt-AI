"""
OmniCourt-AI Sample Video Generator.
Generates two distinct scenarios:
1. runout_sample.mp4  -> OUT (Stumps broken while batter is short of popping crease)
2. not_out_sample.mp4 -> NOT OUT (Batter grounds bat across popping crease before stumps are broken)
"""

import os
import subprocess
import cv2
import numpy as np
import imageio_ffmpeg


def reencode_to_h264(video_path: str):
    """Re-encode video to browser-compliant H.264 (libx264) with yuv420p pixel format."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    temp_path = video_path + ".h264.mp4"
    cmd = [
        ffmpeg_exe, "-y", "-i", video_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        temp_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.replace(temp_path, video_path)


def create_cricket_clip(
    output_path: str,
    scenario: str = "run_out",
    duration_sec: float = 2.5,
    fps: int = 30,
    width: int = 854,
    height: int = 480,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(duration_sec * fps)

    pitch_y = int(height * 0.72)
    crease_x = int(width * 0.55)     # Popping crease line at X=470
    stump_x = int(width * 0.78)      # Stumps at X=666
    stump_bottom_y = pitch_y
    stump_top_y = pitch_y - 110

    # In RUN OUT: Stumps break at T=1.50s (Frame 45) before batter reaches crease
    # In NOT OUT: Batter reaches and grounds at T=1.30s (Frame 39), stumps break later at T=1.80s (Frame 54)
    if scenario == "run_out":
        break_frame = int(1.50 * fps)
    else:
        break_frame = int(1.80 * fps)

    # Static pitch canvas
    pitch_background = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(pitch_y):
        ratio = y / pitch_y
        pitch_background[y, :] = (int(40 + ratio * 20), int(35 + ratio * 25), int(25 + ratio * 20))

    pitch_background[pitch_y:, :] = (34, 110, 45)

    pitch_pts = np.array([
        [int(width * 0.05), pitch_y + 80],
        [int(width * 0.95), pitch_y + 80],
        [int(width * 0.90), pitch_y - 30],
        [int(width * 0.10), pitch_y - 30]
    ], np.int32)
    cv2.fillPoly(pitch_background, [pitch_pts], (140, 180, 205))

    # White popping crease line
    cv2.line(pitch_background, (crease_x, pitch_y - 30), (crease_x, pitch_y + 70), (255, 255, 255), 4)
    cv2.putText(pitch_background, "POPPING CREASE", (crease_x - 80, pitch_y + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    # Stumps
    stump_color = (60, 140, 200)
    for offset in [-12, 0, 12]:
        cv2.line(pitch_background, (stump_x + offset, stump_bottom_y), (stump_x + offset, stump_top_y), stump_color, 4)

    ground_y = pitch_y + 35

    for f in range(total_frames):
        t = f / fps
        frame = pitch_background.copy()

        # 1. Stumps & Bails
        if f < break_frame:
            cv2.line(frame, (stump_x - 16, stump_top_y - 2), (stump_x + 16, stump_top_y - 2), (40, 200, 240), 4)
        else:
            prog = (f - break_frame) / 8.0
            bail1_x = int(stump_x - 16 - prog * 24)
            bail1_y = int(stump_top_y - 2 - prog * 35 + 0.5 * 9.8 * (prog**2) * 12)
            bail2_x = int(stump_x + 16 + prog * 26)
            bail2_y = int(stump_top_y - 2 - prog * 25 + 0.5 * 9.8 * (prog**2) * 12)

            cv2.circle(frame, (stump_x, stump_top_y), int(14 + prog * 8), (0, 140, 255), 2)
            cv2.line(frame, (bail1_x - 6, bail1_y), (bail1_x + 6, bail1_y - 4), (0, 100, 255), 4)
            cv2.line(frame, (bail2_x - 6, bail2_y), (bail2_x + 6, bail2_y + 4), (0, 100, 255), 4)
            cv2.putText(frame, "BAILS DISLODGED", (stump_x - 60, stump_top_y - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 140, 255), 2, cv2.LINE_AA)

        # 2. Ball Flight
        ball_start_x = width - 35
        ball_start_y = int(height * 0.28)
        if f <= break_frame:
            p = f / break_frame
            ball_x = int(ball_start_x + (stump_x - ball_start_x) * p)
            ball_y = int(ball_start_y + (stump_top_y + 12 - ball_start_y) * p)
            cv2.circle(frame, (ball_x, ball_y), 7, (20, 20, 220), -1)
            cv2.circle(frame, (ball_x, ball_y), 8, (255, 255, 255), 1)
        else:
            post = (f - break_frame) / 8.0
            ball_x = int(stump_x - post * 14)
            ball_y = int(stump_top_y + 20 + post * 30)
            cv2.circle(frame, (ball_x, ball_y), 7, (20, 20, 220), -1)

        # 3. Batter Kinematics
        if scenario == "run_out":
            # Batter is 60px SHORT of the crease line when stumps break at T=1.50s
            bat_tip_start_x = int(width * 0.12)
            bat_tip_impact_x = crease_x - 60
            bat_tip_end_x = crease_x + 20

            if f <= break_frame:
                current_tip_x = int(bat_tip_start_x + (bat_tip_impact_x - bat_tip_start_x) * (f / break_frame))
            else:
                rem = max(1, total_frames - break_frame)
                current_tip_x = int(bat_tip_impact_x + (bat_tip_end_x - bat_tip_impact_x) * ((f - break_frame) / rem))

            bat_tip_y = ground_y
            batter_x = current_tip_x - 90
            batter_y = bat_tip_y - 80

            cv2.ellipse(frame, (batter_x + 20, batter_y + 40), (28, 50), -30, 0, 360, (180, 80, 20), -1)
            cv2.circle(frame, (batter_x + 45, batter_y), 15, (220, 200, 180), -1)
            cv2.line(frame, (batter_x + 35, batter_y + 35), (current_tip_x - 45, bat_tip_y - 20), (200, 200, 200), 5)
            cv2.line(frame, (current_tip_x - 45, bat_tip_y - 20), (current_tip_x, bat_tip_y), (30, 130, 210), 6)
            cv2.circle(frame, (current_tip_x, bat_tip_y), 4, (0, 0, 255), -1)

            if f >= break_frame:
                cv2.putText(frame, "SHORT OF CREASE (OUT)", (current_tip_x - 110, batter_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        else:
            # Batter safely crosses crease at T=1.30s; firmly grounded past line when bails break at T=1.80s
            reach_frame = int(1.30 * fps)
            bat_tip_start_x = int(width * 0.15)
            bat_tip_safe_x = crease_x + 60
            bat_tip_end_x = crease_x + 95

            if f <= reach_frame:
                current_tip_x = int(bat_tip_start_x + (bat_tip_safe_x - bat_tip_start_x) * (f / reach_frame))
            else:
                rem = max(1, total_frames - reach_frame)
                current_tip_x = int(bat_tip_safe_x + (bat_tip_end_x - bat_tip_safe_x) * ((f - reach_frame) / rem))

            bat_tip_y = ground_y
            batter_x = current_tip_x - 90
            batter_y = ground_y - 80

            cv2.ellipse(frame, (batter_x + 20, batter_y + 40), (28, 50), -30, 0, 360, (20, 120, 180), -1)
            cv2.circle(frame, (batter_x + 45, batter_y), 15, (220, 200, 180), -1)
            cv2.line(frame, (batter_x + 35, batter_y + 35), (current_tip_x - 45, bat_tip_y - 20), (200, 200, 200), 5)
            cv2.line(frame, (current_tip_x - 45, bat_tip_y - 20), (current_tip_x, bat_tip_y), (30, 130, 210), 6)
            cv2.circle(frame, (current_tip_x, bat_tip_y), 5, (0, 255, 0), -1)

            if current_tip_x > crease_x:
                cv2.putText(frame, "BAT SAFELY GROUNDED (NOT OUT)", (current_tip_x - 130, batter_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        # Broadcast telemetry HUD
        cv2.rectangle(frame, (0, 0), (width, 38), (15, 15, 20), -1)
        cv2.putText(frame, "OMNICOURT DRS BROADCAST FEED", (18, 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 215, 255), 1, cv2.LINE_AA)

        time_text = f"T: {t:.2f}s | Frame {f+1}/{total_frames} (30 FPS)"
        cv2.putText(frame, time_text, (width - 310, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 220, 220), 1, cv2.LINE_AA)

        out.write(frame)

    out.release()
    reencode_to_h264(output_path)
    print(f"Generated H.264 clip at: {output_path}")


if __name__ == "__main__":
    from cv_engine import extract_video_frames

    runout_path = "assets/videos/runout_sample.mp4"
    notout_path = "assets/videos/not_out_sample.mp4"

    create_cricket_clip(runout_path, scenario="run_out")
    create_cricket_clip(notout_path, scenario="not_out")

    # Clean old cache and extract fresh frame sequences
    print("Extracting sample frames into cache...")
    extract_video_frames(runout_path, output_dir="assets/frames/runout_sample", fps_sample=15.0)
    extract_video_frames(notout_path, output_dir="assets/frames/not_out_sample", fps_sample=15.0)
    print("Generation complete! Only runout_sample.mp4 and not_out_sample.mp4 are active.")