import time
import cv2
import numpy as np
import mediapipe as mp

from signal_utils import (
    pos_algorithm, bandpass_filter, estimate_hr_and_hrv,
    eye_aspect_ratio, mouth_aspect_ratio, calculate_comprehensive_stress
)

SCAN_DURATION_SEC = 10.0

# FaceMesh Landmark Indices
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 13, 291, 14]
LEFT_BROW_INNER = 107
RIGHT_BROW_INNER = 336
NOSE_TIP = 1


def get_pts(landmarks, indices, w, h):
    return np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])


def run_10s_stress_scan():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.6)

    print("\n=======================================================")
    print(f" Hold still and look directly at the webcam for {int(SCAN_DURATION_SEC)}s...")
    print("=======================================================\n")

    # Buffers for 10-second collection
    rgb_means = []
    timestamps = []
    ear_history = []
    brow_ratios = []
    head_rotations = []

    start_time = None
    blink_count = 0
    eye_closed = False

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
                start_time = now  # Initialize scan on first face lock

            elapsed = now - start_time
            remaining = max(0.0, SCAN_DURATION_SEC - elapsed)
            landmarks = results.multi_face_landmarks[0].landmark

            # 1. Forehead rPPG Region of Interest
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            x1, x2 = int(min(xs) + 0.30 * (max(xs) - min(xs))), int(min(xs) + 0.70 * (max(xs) - min(xs)))
            y1, y2 = int(min(ys) + 0.08 * (max(ys) - min(ys))), int(min(ys) + 0.28 * (max(ys) - min(ys)))
            
            forehead_roi = rgb_frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if forehead_roi.size > 0:
                rgb_means.append(forehead_roi.reshape(-1, 3).mean(axis=0))
                timestamps.append(now)

            # 2. Eye Aspect Ratio & Blink Detection
            r_eye = get_pts(landmarks, RIGHT_EYE, w, h)
            l_eye = get_pts(landmarks, LEFT_EYE, w, h)
            ear = (eye_aspect_ratio(r_eye) + eye_aspect_ratio(l_eye)) / 2.0
            ear_history.append(ear)

            if ear < 0.21:
                if not eye_closed:
                    blink_count += 1
                    eye_closed = True
            else:
                eye_closed = False

            # 3. Facial Micro-Tension (Brow Furrowing)
            brow_dist = np.linalg.norm(
                np.array([landmarks[LEFT_BROW_INNER].x * w, landmarks[LEFT_BROW_INNER].y * h]) -
                np.array([landmarks[RIGHT_BROW_INNER].x * w, landmarks[RIGHT_BROW_INNER].y * h])
            )
            face_width = max(xs) - min(xs)
            brow_ratios.append(brow_dist / max(face_width, 1.0))

            # 4. Head Jitter (Nose displacement proxy)
            head_rotations.append([landmarks[NOSE_TIP].x * w, landmarks[NOSE_TIP].y * h])

            # Visual Feedback Overlay
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            progress = int((elapsed / SCAN_DURATION_SEC) * w)
            cv2.rectangle(frame, (0, h - 20), (progress, h), (0, 255, 255), -1)
            cv2.putText(frame, f"Scanning: {remaining:.1f}s remaining", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Keep still and look at camera", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            if elapsed >= SCAN_DURATION_SEC:
                break
        else:
            cv2.putText(frame, "No face detected - repositioning...", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("10-Second Stress Assessment", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # --- Processing Results ---
    if len(rgb_means) < 30:
        print("Scan failed: Insufficient frame data collected.")
        return

    actual_fps = len(timestamps) / (timestamps[-1] - timestamps[0])
    pulse = pos_algorithm(np.array(rgb_means), actual_fps)
    filtered_pulse = bandpass_filter(pulse, actual_fps)
    hrv_data = estimate_hr_and_hrv(filtered_pulse, actual_fps)

    # Compute Feature Metrics
    blink_rate_min = (blink_count / SCAN_DURATION_SEC) * 60.0
    mean_brow_ratio = float(np.mean(brow_ratios)) if brow_ratios else 0.20
    head_jitter = float(np.std(head_rotations)) if head_rotations else 1.0

    stress_score, subscores = calculate_comprehensive_stress(
        hrv_data["rmssd_ms"], blink_rate_min, mean_brow_ratio, head_jitter
    )

    # Display Report
    print("\n" + "="*45)
    print("       10-SECOND STRESS SCAN RESULTS")
    print("="*45)
    print(f" Heart Rate (HR)         : {hrv_data['hr_bpm'] if hrv_data['hr_bpm'] else 'Measuring/Calibrating'} BPM")
    print(f" HRV (RMSSD)             : {hrv_data['rmssd_ms'] if hrv_data['rmssd_ms'] else 'Measuring/Calibrating'} ms")
    print(f" Blink Rate              : {blink_rate_min:.1f} blinks/min (Total: {blink_count})")
    print(f" Brow Tension Ratio      : {mean_brow_ratio:.3f}")
    print(f" Head Motion Jitter      : {head_jitter:.2f} px")
    print("-" * 45)
    print(f" OVERALL STRESS SCORE    : {stress_score} / 100")
    if stress_score < 35:
        print(" State Assessment        : CALM / RELAXED")
    elif stress_score < 65:
        print(" State Assessment        : MODERATE / NORMAL FOCUS")
    else:
        print(" State Assessment        : HIGH PHYSIOLOGICAL STRAIN")
    print("="*45 + "\n")


if __name__ == "__main__":
    run_10s_stress_scan()
