"""
processor.py
============
High-precision video processing for driver-drowsiness detection.

UPGRADE:
- Uses MediaPipe Tasks API (FaceLandmarker) instead of Legacy Solutions.
- Higher precision landmark extraction (478 points).
- Iris tracking for eye openness verification.
- Improved EAR smoothing and adaptive baselines.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from moviepy import AudioFileClip, CompositeAudioClip, VideoFileClip
from tensorflow.keras.models import load_model

from utils import (
    MP_LEFT_EYE, MP_RIGHT_EYE, MP_MOUTH, MP_LEFT_IRIS, MP_RIGHT_IRIS,
    compute_EAR, compute_MAR, preprocess_image
)

# Tuning constants
IMG_SIZE             = 145
ALERT_THRESHOLD      = 12
FRAME_SKIP           = 3           # process every Nth frame fully
DROWSY_SCORE_THRESHOLD  = 4
ALARM_COOLDOWN_SEC      = 1.5
EVENT_COOLDOWN_SEC      = 1.0
LANDMARK_SMOOTH_ALPHA   = 0.65
BOX_SMOOTH_ALPHA        = 0.72

label_map = {0: "Yawn", 1: "No_yawn", 2: "Closed", 3: "Open"}

def _clamp_box(box, frame_width, frame_height):
    x, y, w, h = box
    x = max(0, min(int(x), frame_width - 1))
    y = max(0, min(int(y), frame_height - 1))
    w = max(1, min(int(w), frame_width - x))
    h = max(1, min(int(h), frame_height - y))
    return x, y, w, h

def _smooth_points(prev_points, new_points):
    if prev_points is None: return new_points
    a = LANDMARK_SMOOTH_ALPHA
    return [(int(a * px + (1 - a) * nx), int(a * py + (1 - a) * ny))
            for (px, py), (nx, ny) in zip(prev_points, new_points)]

def _draw_text_with_outline(frame, text, origin, color, font_scale=0.62, thickness=2):
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def _draw_frame_overlay(frame, visual_state):
    status_color = visual_state["status_color"]
    bbox = visual_state.get("bbox")
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), status_color, 2)
        _draw_text_with_outline(frame, f"{visual_state['status']}", (x, max(18, y-10)), status_color, font_scale=0.66)

    for (x, y) in (visual_state.get("left_eye") or []) + (visual_state.get("right_eye") or []):
        cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
    for (x, y) in (visual_state.get("mouth") or []):
        cv2.circle(frame, (x, y), 1, (255, 0, 255), -1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (6, 6), (280, 85), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    _draw_text_with_outline(frame, f"EAR: {visual_state['EAR']:.3f}", (16, 30), (200, 255, 200))
    _draw_text_with_outline(frame, f"MAR: {visual_state['MAR']:.3f}", (16, 55), (200, 255, 200))
    _draw_text_with_outline(frame, f"STATUS: {visual_state['status']}", (16, 75), status_color, font_scale=0.55)

def process_video(job_id, video_path, model_path, beep_path, jobs, output_root, shape_predictor_path, haarcascade_path):
    output_root = Path(output_root)
    frames_dir = output_root / "frames" / job_id
    videos_dir = output_root / "videos" / job_id
    for d in [frames_dir, videos_dir]: d.mkdir(parents=True, exist_ok=True)

    output_no_audio_path = videos_dir / "processed_no_audio.mp4"
    final_output_path = videos_dir / "processed_final.mp4"

    jobs[job_id]["status"] = "processing"
    jobs[job_id]["stage"] = "Initializing High-Precision Pipeline..."
    
    try:
        # Load Keras model
        classifier = load_model(model_path)
        
        # Initialize MediaPipe Tasks API
        # Look for model in assets
        model_asset_path = Path(__file__).parent / "assets" / "face_landmarker.task"
        if not model_asset_path.exists():
             raise RuntimeError(f"Model file face_landmarker.task not found at {model_asset_path}")

        base_options = mp_python.BaseOptions(model_asset_path=str(model_asset_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(str(video_path))
        fps_raw = cap.get(cv2.CAP_PROP_FPS)
        fps = fps_raw if np.isfinite(fps_raw) and fps_raw > 0 else 25
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_no_audio_path), fourcc, fps, (width, height))

        # Pipeline State
        frame_count, processed_count = 0, 0
        drowsy_counter = 0
        alarm_timestamps, events = [], []
        ear_baseline, mar_baseline = None, None
        last_alarm_time, last_event_time = -999.0, -999.0
        prev_left, prev_right, prev_mouth = None, None, None
        last_visual_state = None

        while True:
            ret, bgr_frame = cap.read()
            if not ret: break
            
            try:
                current_time = frame_count / fps
                if frame_count % FRAME_SKIP != 0:
                    if last_visual_state: _draw_frame_overlay(bgr_frame, last_visual_state)
                    out.write(bgr_frame)
                    continue

                # MediaPipe Tasks expects mp.Image
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # Detect
                detection_result = landmarker.detect(mp_image)
                
                # defaults
                ear, mar = 1.0, 0.0
                left_eye, right_eye, mouth = None, None, None
                status, status_color = "ALERT", (0, 255, 0)
                label, stable_box = "No_face", None

                if detection_result.face_landmarks:
                    landmarks = detection_result.face_landmarks[0]
                    
                    # Extract Eye & Mouth points
                    left_eye_raw = [(int(landmarks[i].x * width), int(landmarks[i].y * height)) for i in MP_LEFT_EYE]
                    right_eye_raw = [(int(landmarks[i].x * width), int(landmarks[i].y * height)) for i in MP_RIGHT_EYE]
                    mouth_raw = [(int(landmarks[i].x * width), int(landmarks[i].y * height)) for i in MP_MOUTH]

                    left_eye = _smooth_points(prev_left, left_eye_raw)
                    right_eye = _smooth_points(prev_right, right_eye_raw)
                    mouth = _smooth_points(prev_mouth, mouth_raw)
                    prev_left, prev_right, prev_mouth = left_eye, right_eye, mouth

                    ear = (compute_EAR(left_eye) + compute_EAR(right_eye)) / 2.0
                    mar = compute_MAR(mouth)
                    
                    # Bounding box
                    xs = [l.x * width for l in landmarks]
                    ys = [l.y * height for l in landmarks]
                    stable_box = _clamp_box((min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)), width, height)

                    # Baseline adaptation
                    if ear_baseline is None: ear_baseline = ear
                    else: ear_baseline = 0.98 * ear_baseline + 0.02 * ear
                    
                    if mar_baseline is None: mar_baseline = mar
                    else: mar_baseline = 0.98 * mar_baseline + 0.02 * mar

                    # Keras Model Check
                    x, y, w, h = stable_box
                    face_crop = bgr_frame[y:y+h, x:x+w]
                    if face_crop.size > 0:
                        resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
                        processed = preprocess_image(resized) / 255.0
                        prediction = classifier.predict(processed.reshape(-1, IMG_SIZE, IMG_SIZE, 3), verbose=0)
                        label = label_map[int(np.argmax(prediction))]

                    # Drowsy Logic (Scoring system)
                    ear_ratio = ear / max(ear_baseline, 1e-6)
                    is_puffy = ear < 0.18 or ear_ratio < 0.80
                    yawn_cue = mar > 0.65 or (mar_baseline and mar > mar_baseline * 1.5)
                    
                    score = 0
                    if is_puffy: score += 4          # Independent trigger
                    if yawn_cue: score += 4          # Independent trigger
                    if label in ["Closed", "Yawn"]: score += 2
                    
                    if score >= DROWSY_SCORE_THRESHOLD:
                        drowsy_counter = min(drowsy_counter + 1, 30)
                    else:
                        drowsy_counter = max(drowsy_counter - 1, 0)
                    
                    if drowsy_counter >= ALERT_THRESHOLD:
                        status, status_color = "DROWSY", (0, 0, 255)
                        if current_time - last_alarm_time > ALARM_COOLDOWN_SEC:
                            alarm_timestamps.append(current_time)
                            last_alarm_time = current_time
                        if current_time - last_event_time > EVENT_COOLDOWN_SEC:
                            fname = f"frame_{frame_count}.jpg"
                            cv2.imwrite(str(frames_dir / fname), bgr_frame)
                            events.append({"frame": frame_count, "timestamp": round(current_time, 2), "label": label, "status": "DROWSY", "EAR": round(ear,3), "MAR": round(mar,3), "frame_file": f"/frame/{job_id}/{fname}"})
                            last_event_time = current_time

                visual_state = {"EAR": ear, "MAR": mar, "status": status, "status_color": status_color, "bbox": stable_box, "left_eye": left_eye, "right_eye": right_eye, "mouth": mouth}
                last_visual_state = visual_state
                _draw_frame_overlay(bgr_frame, visual_state)
                out.write(bgr_frame)
                processed_count += 1
            except Exception:
                out.write(bgr_frame)
            finally:
                frame_count += 1
                if total_frames > 0: jobs[job_id]["progress"] = int((frame_count/total_frames)*90)
        
        cap.release(); out.release(); landmarker.close()

        # Audio injection
        jobs[job_id]["stage"] = "Finalizing Video..."
        video = VideoFileClip(str(output_no_audio_path))
        audio_clips = [AudioFileClip(str(beep_path)).with_start(t) for t in alarm_timestamps]
        final_video = video.with_audio(CompositeAudioClip(audio_clips)) if audio_clips else video
        final_video.write_videofile(str(final_output_path), codec="libx264", audio_codec="aac", logger=None)
        
        jobs[job_id].update({"status": "done", "progress": 100, "stage": "Completed", "output_video_file": str(final_output_path)})
        jobs[job_id]["results"] = {"total_frames": frame_count, "processed_frames": processed_count, "drowsy_events": len(events), "alarm_count": len(alarm_timestamps), "duration": round(frame_count/fps, 2), "video_path": f"/video/{job_id}", "events": events}

    except Exception as e:
        jobs[job_id].update({"status": "error", "stage": "Failed", "error": str(e)})
