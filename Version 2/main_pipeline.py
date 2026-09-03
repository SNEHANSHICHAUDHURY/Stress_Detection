import os
import time
import json
import numpy as np
import cv2
import mediapipe as mp
import sounddevice as sd
import threading
import random
from google import genai
from google.genai import types 

from pipeline_utils import (
    pos_algorithm, bandpass_filter, estimate_hr_and_hrv,
    eye_aspect_ratio, extract_voice_stress_features
)

# Configuration
BASELINE_DURATION_SEC = 15.0
AUDIO_SAMPLE_RATE = 22050
AUDIO_RECORD_DURATION = 10.0  # Time candidate speaks their answer

# Landmark indices
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_BROW = 107
RIGHT_BROW = 336
NOSE_TIP = 1


def get_pts(landmarks, indices, w, h):
    return np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])


# ---------------------------------------------------------
# STAGE 1: Silent 15-Second Physiological Baseline
# ---------------------------------------------------------
def run_stage_1_baseline(cap, face_mesh):
    print("\n" + "="*50)
    print(f" STAGE 1: SILENT BASELINE ({int(BASELINE_DURATION_SEC)}s)")
    print(" Please sit still, remain silent, and look at the camera.")
    print("="*50 + "\n")

    rgb_means = []
    timestamps = []
    ear_list = []
    brow_ratios = []
    head_jitters = []
    blink_count = 0
    eye_closed = False

    start_time = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            if start_time is None:
                start_time = now

            elapsed = now - start_time
            landmarks = results.multi_face_landmarks[0].landmark

            # 1. Forehead rPPG
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            x1, x2 = int(min(xs) + 0.30 * (max(xs) - min(xs))), int(min(xs) + 0.70 * (max(xs) - min(xs)))
            y1, y2 = int(min(ys) + 0.08 * (max(ys) - min(ys))), int(min(ys) + 0.28 * (max(ys) - min(ys)))
            
            forehead = rgb_frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if forehead.size > 0:
                rgb_means.append(forehead.reshape(-1, 3).mean(axis=0))
                timestamps.append(now)

            # 2. Blink & Tension Tracking
            r_eye = get_pts(landmarks, RIGHT_EYE, w, h)
            l_eye = get_pts(landmarks, LEFT_EYE, w, h)
            ear = (eye_aspect_ratio(r_eye) + eye_aspect_ratio(l_eye)) / 2.0
            ear_list.append(ear)

            if ear < 0.21:
                if not eye_closed:
                    blink_count += 1
                    eye_closed = True
            else:
                eye_closed = False

            brow_d = np.linalg.norm(
                np.array([landmarks[LEFT_BROW].x * w, landmarks[LEFT_BROW].y * h]) -
                np.array([landmarks[RIGHT_BROW].x * w, landmarks[RIGHT_BROW].y * h])
            )
            brow_ratios.append(brow_d / max(max(xs) - min(xs), 1.0))
            head_jitters.append([landmarks[NOSE_TIP].x * w, landmarks[NOSE_TIP].y * h])

            # UI Overlay
            remaining = max(0.0, BASELINE_DURATION_SEC - elapsed)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Stage 1 Baseline: {remaining:.1f}s remaining", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if elapsed >= BASELINE_DURATION_SEC:
                break
        else:
            cv2.putText(frame, "Align Face with Camera...", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Multi-Modal Stress Assessment", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calculate Stage 1 Results
    actual_fps = len(timestamps) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 30.0
    pulse = pos_algorithm(np.array(rgb_means), actual_fps)
    filtered = bandpass_filter(pulse, actual_fps)
    hrv = estimate_hr_and_hrv(filtered, actual_fps)

    # RMSSD to Stress Sub-score (0-100)
    rmssd = hrv["rmssd_ms"] if hrv["rmssd_ms"] is not None else 50.0
    s_hrv = (1.0 - (np.clip(rmssd, 20.0, 80.0) - 20.0) / 60.0) * 100.0

    return {
        "hr_bpm": hrv["hr_bpm"] or 72.0,
        "rmssd_ms": hrv["rmssd_ms"] or 45.0,
        "blink_rate": (blink_count / BASELINE_DURATION_SEC) * 60.0,
        "brow_ratio": float(np.mean(brow_ratios)) if brow_ratios else 0.22,
        "head_jitter": float(np.std(head_jitters)) if head_jitters else 1.0,
        "s_hrv_subscore": round(float(s_hrv), 1)
    }


# ---------------------------------------------------------
# STAGE 2A: Gemini Adaptive Question Generator
# ---------------------------------------------------------
def generate_adaptive_prompt(baseline_data):
    """
    Calls the Gemini API to produce an interview question based on the baseline
    and a randomly selected psychological dimension.
    """
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Randomize the assessment focus for variety
    topics = [
        "recent sleep quality and physical recovery",
        "current cognitive workload and focus",
        "recent feelings of being rushed or overwhelmed",
        "ability to disconnect and relax after a long day",
        "recent changes in appetite or daily routine"
    ]
    chosen_topic = random.choice(topics)

    prompt = f"""
    You are an objective clinical screening assistant. 
    The candidate has completed their Stage 1 baseline scan:
    - Resting HR: {baseline_data['hr_bpm']} BPM
    - HRV (RMSSD): {baseline_data['rmssd_ms']} ms (Lower = higher strain)

    Generate exactly ONE targeted question to assess their {chosen_topic}.
    Keep the question conversational, under 20 words, and do not use standard greetings.

    Return JSON with this exact schema:
    {{
       "question_text": "...",
       "focus_area": "{chosen_topic}"
    }}
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.8  # Increased from 0.2 to encourage phrasing variety
            )
        )
        parsed = json.loads(response.text)
        return parsed["question_text"]
    except Exception as e:
        # If API fails, print error and return a generic fallback
        print("\n" + "!"*55)
        print(" GEMINI API ERROR TRIGGERED:")
        print(f" {type(e).__name__}: {str(e)}")
        print("!"*55 + "\n")
        return "Tell me about a recent situation that required your intense focus, and how you managed it."

# ---------------------------------------------------------
# STAGE 2B: Audio Recording & Background Vision Tracking
# ---------------------------------------------------------
def run_stage_2_recording_and_vision(cap, face_mesh, question_text):
    print("\n" + "="*50)
    print(f" STAGE 2: ADAPTIVE INTERVIEW")
    print(f" QUESTION: \"{question_text}\"")
    print(f" Please speak your answer clearly, then press ENTER in the video window...")
    print("="*50 + "\n")

    # 1. Setup continuous background audio streaming
    audio_buffer = []
    
    def audio_callback(indata, frames, time_info, status):
        # This function is called continuously by sounddevice to append audio chunks
        audio_buffer.extend(indata.copy())

    stream = sd.InputStream(samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype='float32', callback=audio_callback)
    stream.start()

    start_time = time.time()
    ear_list = []
    blink_count = 0
    eye_closed = False

    # 2. Continuous Video Loop until ENTER (ASCII 13) is pressed
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()
        elapsed = now - start_time

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            r_eye = get_pts(landmarks, RIGHT_EYE, w, h)
            l_eye = get_pts(landmarks, LEFT_EYE, w, h)
            ear = (eye_aspect_ratio(r_eye) + eye_aspect_ratio(l_eye)) / 2.0
            ear_list.append(ear)

            if ear < 0.21:
                if not eye_closed:
                    blink_count += 1
                    eye_closed = True
            else:
                eye_closed = False

        # Visual Prompt & Dynamic UI
        cv2.putText(frame, "STAGE 2: SPEAK YOUR ANSWER", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Blink text to show it is actively recording
        if int(elapsed * 2) % 2 == 0:
            cv2.putText(frame, f"[RECORDING] Elapsed: {elapsed:.1f}s", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, "Press ENTER when finished", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Word wrap question on screen
        cv2.putText(frame, question_text[:50], (30, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        if len(question_text) > 50:
            cv2.putText(frame, question_text[50:100], (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow("Multi-Modal Stress Assessment", frame)
        
        # Wait for the Enter key (ASCII code 13)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            print("\nEnter pressed. Finalizing recording...")
            break

    # 3. Stop stream and process data
    stream.stop()
    stream.close()
    
    # Flatten the audio buffer into a 1D array for librosa
    audio_flat = np.array(audio_buffer).flatten() if audio_buffer else np.zeros(AUDIO_SAMPLE_RATE)
    voice_metrics = extract_voice_stress_features(audio_flat, sr=AUDIO_SAMPLE_RATE)

    # Normalize blink rate based on however long they decided to talk
    actual_duration = time.time() - start_time
    stage_2_blink_rate = (blink_count / actual_duration) * 60.0 if actual_duration > 0 else 0.0
    
    return voice_metrics, stage_2_blink_rate


# ---------------------------------------------------------
# STAGE 3: Multi-Modal Decision Fusion
# ---------------------------------------------------------
def fuse_multimodal_scores(baseline, voice, stage2_blink_rate):
    """
    Combines:
    - 40% Physiological Baseline (HRV / RMSSD)
    - 30% Voice Acoustic Stress (Pitch variation & spectral energy)
    - 30% Behavioral Tension (Blink elevation & brow furrowing)
    """
    s_hrv = baseline["s_hrv_subscore"]
    s_voice = voice["vocal_stress_subscore"]

    # Behavioral subscore
    s_blink = np.clip((stage2_blink_rate - 14.0) / (32.0 - 14.0) * 100.0, 0, 100)
    s_brow = np.clip((0.22 - baseline["brow_ratio"]) / (0.22 - 0.14) * 100.0, 0, 100)
    s_behavior = 0.60 * s_blink + 0.40 * s_brow

    total_stress = (0.40 * s_hrv) + (0.30 * s_voice) + (0.30 * s_behavior)
    return round(float(total_stress), 1), {
        "hrv_subscore": s_hrv,
        "voice_subscore": s_voice,
        "behavior_subscore": round(float(s_behavior), 1)
    }


# ---------------------------------------------------------
# Main Execution Flow
# ---------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera access failed.")
        return

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.6) as face_mesh:
        
        # 1. Run Baseline Scan
        baseline = run_stage_1_baseline(cap, face_mesh)
        print(f"\nBaseline Calculated: HR = {baseline['hr_bpm']} BPM | RMSSD = {baseline['rmssd_ms']} ms")

        # 2. Fetch Question in Background so Video Doesn't Freeze
        print("\nGenerating adaptive question from physiological state...")
        fetched_question = [""] # Use a list to store the result from the thread
        
        def fetch_q():
            fetched_question[0] = generate_adaptive_prompt(baseline)
            
        q_thread = threading.Thread(target=fetch_q)
        q_thread.start()
        
        # Keep updating the video feed while Gemini is thinking
        while q_thread.is_alive():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, "Analyzing Baseline & Generating Question...", (30, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Multi-Modal Stress Assessment", frame)
            cv2.waitKey(1)

        # 3. Interactive Voice + Vision Recording
        voice_results, s2_blinks = run_stage_2_recording_and_vision(cap, face_mesh, fetched_question[0])

    cap.release()
    cv2.destroyAllWindows()

    # 4. Final Fusion
    overall_score, subscores = fuse_multimodal_scores(baseline, voice_results, s2_blinks)

    # 5. Output Summary Report
    print("\n" + "="*55)
    print("      FINAL MULTI-MODAL STRESS ASSESSMENT")
    print("="*55)
    print(f" [PHYSIOLOGY] Resting Heart Rate   : {baseline['hr_bpm']} BPM")
    print(f" [PHYSIOLOGY] Resting HRV (RMSSD)  : {baseline['rmssd_ms']} ms")
    print(f" [ACOUSTICS]  Mean Voice Pitch (F0): {voice_results['pitch_mean_hz']} Hz (±{voice_results['pitch_std_hz']} Hz)")
    print(f" [BEHAVIOR]   Speech Blink Rate    : {s2_blinks:.1f} blinks/min")
    print("-" * 55)
    print(f" Sub-Scores: HRV={subscores['hrv_subscore']} | Voice={subscores['voice_subscore']} | Behavior={subscores['behavior_subscore']}")
    print(f" OVERALL STRESS INDEX              : {overall_score} / 100")
    
    if overall_score < 35:
        print(" CLINICAL STATUS                   : LOW STRESS / STABLE")
    elif overall_score < 65:
        print(" CLINICAL STATUS                   : MODERATE LOAD / ELEVATED")
    else:
        print(" CLINICAL STATUS                   : HIGH ACUTE STRAIN")
    print("="*55 + "\n")

if __name__ == "__main__":
    main()