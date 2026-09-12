import os
import glob
import numpy as np
import cv2 as cv
import pandas as pd
import mediapipe as mp
import librosa
from moviepy.video.io.VideoFileClip import VideoFileClip
from pathlib import Path
from pipeline_utils import (pos_algorithm, bandpass_filter, estimate_hr_and_hrv, eye_aspect_ratio, extract_voice_stress_features)

DATASET_ROOT="Video_Dataset"

def extract_audio_from_video(video_path, sr=22050):
    clip=None
    try:
        clip=VideoFileClip(video_path)
        if clip.audio is None:
            return np.zeros(sr*3)

        audio_array=clip.audio.to_soundarray(fps=sr)
        if audio_array.ndim>1:
            audio_array=audio_array.mean(axis=1)

        clip.close()
        return audio_array.astype(np.float32)

    except Exception as e:
        print(f"failed to extract audio from {video_path}: {e}")
        if clip is not None:
            clip.close()
        return np.zeros(sr * 3)


def fuse_multimodal_scores(hrv_rmssd, vocal_subscore, brow_ratio, speech_blink_rate):
    """
    Computes fusion subscores and overall stress index matching main_pipeline.py logic.
    """
    # 1. HRV Subscore
    rmssd = hrv_rmssd if hrv_rmssd is not None else 50.0
    s_hrv = (1.0 - (np.clip(rmssd, 20.0, 80.0) - 20.0) / 60.0) * 100.0

    # 2. Voice Subscore
    s_voice = vocal_subscore if vocal_subscore is not None else 0.0

    # 3. Behavioral Subscore
    brow = brow_ratio if brow_ratio is not None else 0.22
    s_blink = np.clip((speech_blink_rate - 14.0) / (32.0 - 14.0) * 100.0, 0, 100)
    s_brow = np.clip((0.22 - brow) / (0.22 - 0.14) * 100.0, 0, 100)
    s_behavior = 0.60 * s_blink + 0.40 * s_brow

    # 4. Overall Stress Index Fusion
    total_stress = (0.40 * s_hrv) + (0.30 * s_voice) + (0.30 * s_behavior)

    return {
        "hrv_subscore": round(float(s_hrv), 2),
        "voice_subscore": round(float(s_voice), 2),
        "behavior_subscore": round(float(s_behavior), 2),
        "overall_stress_index": round(float(total_stress), 2)
    }

def process_single_video(video_path, face_mesh):
    cap=cv.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps=cap.get(cv.CAP_PROP_FPS)
    if fps<=0 or np.isnan(fps):
        fps=30.0
    
    rgb_means=[]
    timestamps=[]
    ear_list=[]
    brow_ratios=[]
    head_jitters=[]
    blink_count=0
    eye_closed=False

    frame_idx=0

    while cap.isOpened():
        isTrue, frame=cap.read()
        if not isTrue:
            break

        h, w=frame.shape[:2]
        now=frame_idx/fps
        rgb_frame=cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results=face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks=results.multi_face_landmarks[0].landmark
            xs=[lm.x * w for lm in landmarks]
            ys=[lm.y * h for lm in landmarks]
            x1, x2 = int(min(xs) + 0.30 * (max(xs) - min(xs))), int(min(xs) + 0.70 * (max(xs) - min(xs)))
            y1, y2 = int(min(ys) + 0.08 * (max(ys) - min(ys))), int(min(ys) + 0.28 * (max(ys) - min(ys)))

            forehead=rgb_frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if forehead.size>0:
                rgb_means.append(forehead.reshape(-1, 3).mean(axis=0))
                timestamps.append(now)

            RIGHT_EYE=[33, 160, 158, 133, 153, 144]
            LEFT_EYE=[362, 385, 387, 263, 373, 380]

            r_eye = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in RIGHT_EYE])
            l_eye = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in LEFT_EYE])
            ear = (eye_aspect_ratio(r_eye) + eye_aspect_ratio(l_eye)) / 2.0
            ear_list.append(ear)

            if ear<0.21:
                if not eye_closed:
                    blink_count += 1
                    eye_closed = True
            else:
                eye_closed = False

            LEFT_BROW, RIGHT_BROW, NOSE_TIP = 107, 336, 1
            brow_d = np.linalg.norm(
                np.array([landmarks[LEFT_BROW].x * w, landmarks[LEFT_BROW].y * h]) -
                np.array([landmarks[RIGHT_BROW].x * w, landmarks[RIGHT_BROW].y * h])
            )
            brow_ratios.append(brow_d / max(max(xs) - min(xs), 1.0))
            head_jitters.append([landmarks[NOSE_TIP].x * w, landmarks[NOSE_TIP].y * h])

        frame_idx += 1
    cap.release()

    duration = frame_idx / fps if fps > 0 else 3.0

    if len(rgb_means) > 15:
        pulse = pos_algorithm(np.array(rgb_means), fps)
        filtered = bandpass_filter(pulse, fps)
        hrv = estimate_hr_and_hrv(filtered, fps)
    else:
        hrv = {"hr_bpm": None, "rmssd_ms": None}

    audio_data = extract_audio_from_video(video_path)
    voice_metrics = extract_voice_stress_features(audio_data, sr=22050)

    # Calculate individual attributes
    hr_bpm = round(float(hrv["hr_bpm"]), 2) if hrv["hr_bpm"] is not None else 72.00
    rmssd_ms = round(float(hrv["rmssd_ms"]), 2) if hrv["rmssd_ms"] is not None else 45.00
    
    calculated_blink_rate = (blink_count / duration) * 60.0 if duration > 0 else 0.0
    baseline_blink_rate = round(float(calculated_blink_rate), 2)
    speech_blink_rate = round(float(calculated_blink_rate), 2)  # For single video dataset clips
    
    brow_ratio = round(float(np.mean(brow_ratios)), 4) if brow_ratios else 0.2200
    head_jitter = round(float(np.std(head_jitters)), 4) if head_jitters else 1.0000
    
    pitch_mean_hz = round(float(voice_metrics["pitch_mean_hz"]), 2) if voice_metrics["pitch_mean_hz"] is not None else 0.00
    pitch_std_hz = round(float(voice_metrics["pitch_std_hz"]), 2) if voice_metrics["pitch_std_hz"] is not None else 0.00

    # Fusion processing
    fused_scores = fuse_multimodal_scores(
        hrv_rmssd=rmssd_ms,
        vocal_subscore=voice_metrics.get("vocal_stress_subscore", 0.0),
        brow_ratio=brow_ratio,
        speech_blink_rate=speech_blink_rate
    )

    return {
        "file_name": Path(video_path).name,
        "actor": Path(video_path).parent.name,
        "duration_sec": round(duration, 2),
        "hr_bpm": hr_bpm,
        "rmssd_ms": rmssd_ms,
        "baseline_blink_rate": baseline_blink_rate,
        "brow_ratio": brow_ratio,
        "head_jitter": head_jitter,
        "pitch_mean_hz": pitch_mean_hz,
        "pitch_std_hz": pitch_std_hz,
        "speech_blink_rate": speech_blink_rate,
        "hrv_subscore": fused_scores["hrv_subscore"],
        "voice_subscore": fused_scores["voice_subscore"],
        "behavior_subscore": fused_scores["behavior_subscore"],
        "overall_stress_index": fused_scores["overall_stress_index"]
    }


def batch_process_dataset():
    video_files = glob.glob(os.path.join(DATASET_ROOT, "**", "*.mp4"), recursive=True)
    print(f"Found {len(video_files)} video clips for processing.")

    results = []
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        for idx, video_path in enumerate(video_files, 1):
            print(f"[{idx}/{len(video_files)}] Processing: {video_path}")
            res = process_single_video(video_path, face_mesh)
            if res:
                results.append(res)

    df = pd.DataFrame(results)
   
    column_order = [
        "file_name", "actor", "duration_sec",
        "hr_bpm", "rmssd_ms", "baseline_blink_rate",
        "brow_ratio", "head_jitter", "pitch_mean_hz",
        "pitch_std_hz", "speech_blink_rate", "hrv_subscore",
        "voice_subscore", "behavior_subscore", "overall_stress_index"
    ]
    
    df = df[column_order]
    df.to_csv("dataset_stress_features.csv", index=False)
    print("\nProcessing complete! Features saved to 'dataset_stress_features.csv'.")


if __name__ == "__main__":
    batch_process_dataset()