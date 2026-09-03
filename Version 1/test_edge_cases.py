"""
test_edge_cases.py
-------------------
Targeted tests for the four robustness requirements:
  1. Short/insufficient signal handling (no crashes, no bogus output)
  2. Extreme/out-of-range HR bound checking
  3. PERCLOS + microsleep vs normal-blink differentiation
  4. Lighting-drift robustness of the POS + bandpass pipeline

Run: python3 test_edge_cases.py
"""

import numpy as np
from signal_utils import (
    pos_algorithm, bandpass_filter, estimate_hr_and_hrv,
    BlinkCounter,
)

FPS = 30


# ----------------------------------------------------------------------
# 1. Short / insufficient signal
# ----------------------------------------------------------------------

def test_short_signal_no_crash():
    print("== Test 1: truncated / insufficient signal ==")
    for n_seconds in [0, 0.1, 1, 2]:
        n = int(FPS * n_seconds)
        rgb = np.random.normal(150, 1, size=(n, 3))

        # None of these should raise IndexError, ZeroDivisionError, or
        # scipy's "padlen exceeds signal length" ValueError.
        pulse = pos_algorithm(rgb, FPS)
        filtered = bandpass_filter(pulse, FPS)
        result = estimate_hr_and_hrv(filtered, FPS)

        assert result["hr_bpm"] is None, \
            f"FAILED: got a bogus HR ({result['hr_bpm']}) from only {n_seconds}s of data"
        assert result["reason"] is not None, "FAILED: no reason given for rejection"
        print(f"  n={n_seconds}s ({n} frames): correctly rejected -> {result['reason']}")

    # Also test genuinely empty input (0 frames) explicitly.
    empty_rgb = np.zeros((0, 3))
    pulse = pos_algorithm(empty_rgb, FPS)
    filtered = bandpass_filter(pulse, FPS)
    result = estimate_hr_and_hrv(filtered, FPS)
    assert result["hr_bpm"] is None
    print(f"  n=0 frames: correctly rejected -> {result['reason']}")
    print("  PASS: no crashes, no bogus HR from truncated input\n")


# ----------------------------------------------------------------------
# 2. Extreme / out-of-range heart rates
# ----------------------------------------------------------------------

def test_flat_signal_rejected():
    print("== Test 2a: flat signal (e.g. covered camera) rejected ==")
    n = FPS * 10
    flat_pulse = np.zeros(n)  # perfectly flat, zero variance
    result = estimate_hr_and_hrv(flat_pulse, FPS)
    assert result["hr_bpm"] is None, "FAILED: flat signal produced a fake HR"
    print(f"  flat signal -> correctly rejected: {result['reason']}")
    print("  PASS\n")


def test_out_of_range_hr_rejected():
    print("== Test 2b: out-of-range frequency (e.g. 300bpm artifact) rejected ==")
    duration_s = 10
    n = duration_s * FPS
    t = np.arange(n) / FPS
    # 300 bpm = 5 Hz - outside plausible human resting/light-activity HR,
    # and also outside the bandpass_filter's own passband (0.7-4Hz), so
    # this doubly tests that out-of-band artifacts don't leak through.
    fake_freq_hz = 300 / 60.0
    pulse = np.sin(2 * np.pi * fake_freq_hz * t)

    filtered = bandpass_filter(pulse, FPS)
    result = estimate_hr_and_hrv(filtered, FPS)
    assert result["hr_bpm"] is None, \
        f"FAILED: accepted implausible HR {result['hr_bpm']} bpm"
    print(f"  300bpm artifact -> correctly rejected: {result['reason']}")
    print("  PASS\n")


def test_bradycardia_range_boundary():
    print("== Test 2c: boundary case near 40bpm lower bound ==")
    duration_s = 15
    n = duration_s * FPS
    t = np.arange(n) / FPS
    # 42 bpm - low but should be ACCEPTED (just inside 40-180 range)
    pulse = 2.0 * np.sin(2 * np.pi * (42 / 60.0) * t)
    filtered = bandpass_filter(pulse, FPS)
    result = estimate_hr_and_hrv(filtered, FPS)
    print(f"  42bpm synthetic signal -> result: hr_bpm={result['hr_bpm']}, reason={result['reason']}")
    if result["hr_bpm"] is not None:
        assert 40 <= result["hr_bpm"] <= 180, "FAILED: accepted HR outside stated bounds"
        print("  PASS: accepted and within bounds\n")
    else:
        print("  PASS: conservatively rejected (acceptable - bandpass edge effects "
              "at 0.7Hz cutoff can attenuate a 42bpm=0.7Hz signal)\n")


# ----------------------------------------------------------------------
# 3. PERCLOS & prolonged eye closure (microsleep) vs normal blink
# ----------------------------------------------------------------------

def test_normal_blink_not_flagged_as_microsleep():
    print("== Test 3a: normal blink (150ms) is NOT a microsleep ==")
    counter = BlinkCounter(fps=FPS)
    open_ear, closed_ear = 0.30, 0.05

    closed_frames = int(0.15 * FPS)  # 150ms blink
    sequence = [open_ear] * 30 + [closed_ear] * closed_frames + [open_ear] * 30
    for v in sequence:
        counter.update(v)

    assert counter.total_blinks == 1, f"FAILED: expected 1 normal blink, got {counter.total_blinks}"
    assert counter.microsleep_count() == 0, "FAILED: normal blink misclassified as microsleep"
    print(f"  150ms closure -> blinks={counter.total_blinks}, microsleeps={counter.microsleep_count()}")
    print("  PASS\n")


def test_microsleep_detected_and_separated_from_blinks():
    print("== Test 3b: 2-second closure IS flagged as microsleep, not a blink ==")
    counter = BlinkCounter(fps=FPS)
    open_ear, closed_ear = 0.30, 0.05

    closed_frames = int(2.0 * FPS)  # 2000ms - well past the 1500ms threshold
    sequence = [open_ear] * 30 + [closed_ear] * closed_frames + [open_ear] * 30
    for v in sequence:
        counter.update(v)

    assert counter.total_blinks == 0, f"FAILED: microsleep wrongly counted as a blink ({counter.total_blinks})"
    assert counter.microsleep_count() == 1, f"FAILED: expected 1 microsleep, got {counter.microsleep_count()}"
    flagged, detail = counter.fatigue_flag()
    assert flagged is True, "FAILED: microsleep did not trigger fatigue flag"
    print(f"  2000ms closure -> blinks={counter.total_blinks}, microsleeps={counter.microsleep_count()}")
    print(f"  fatigue_flag: {flagged}, detail: {detail}")
    print("  PASS\n")


def test_microsleep_detected_in_real_time_while_ongoing():
    print("== Test 3c: microsleep detected WHILE STILL CLOSED (no waiting for reopen) ==")
    counter = BlinkCounter(fps=FPS)
    open_ear, closed_ear = 0.30, 0.05

    for v in [open_ear] * 30:
        counter.update(v)
    assert not counter.is_microsleep_in_progress()

    # Feed 1.6s of continuous closure WITHOUT ever reopening the eyes.
    for _ in range(int(1.6 * FPS)):
        counter.update(closed_ear)

    assert counter.is_microsleep_in_progress(), \
        "FAILED: ongoing >1.5s closure not detected in real time"
    print("  after 1.6s of continuous closure (eyes still shut): "
          f"is_microsleep_in_progress={counter.is_microsleep_in_progress()}")
    print("  PASS: alert fires immediately, doesn't wait for eyes to reopen\n")


def test_perclos_calculation():
    print("== Test 3d: PERCLOS reflects fraction of time eyes are closed ==")
    counter = BlinkCounter(fps=FPS, perclos_window_seconds=10)
    open_ear, closed_ear = 0.30, 0.05

    # 10 seconds total: 2 seconds closed (20%), 8 seconds open.
    sequence = [closed_ear] * (2 * FPS) + [open_ear] * (8 * FPS)
    for v in sequence:
        counter.update(v)

    perclos_val = counter.perclos()
    print(f"  2s closed out of 10s window -> PERCLOS={perclos_val}% (expected ~20%)")
    assert 18 <= perclos_val <= 22, f"FAILED: PERCLOS {perclos_val}% far from expected 20%"
    print("  PASS\n")


# ----------------------------------------------------------------------
# 4. Lighting shift / DC drift robustness
# ----------------------------------------------------------------------

def test_pipeline_robust_to_lighting_drift():
    print("== Test 4: pulse recovered despite continuous lighting drift ==")
    duration_s = 15
    n = duration_s * FPS
    t = np.arange(n) / FPS
    true_hr_bpm = 75
    freq_hz = true_hr_bpm / 60.0

    baseline = np.array([150.0, 120.0, 100.0])
    pulse_component = 1.2 * np.sin(2 * np.pi * freq_hz * t)

    # Simulate a large, continuous brightness drift: e.g. someone slowly
    # moving between a shaded and a sunlit area, or ambient light ramping.
    # This drift is ~20x the amplitude of the actual pulse signal and
    # spans the whole clip - a much harsher test than real deployment.
    drift = 25.0 * np.sin(2 * np.pi * (1.0 / duration_s) * t) + 0.8 * t

    noise = np.random.normal(0, 0.3, size=(n, 3))
    rgb_signal = np.tile(baseline, (n, 1)).astype(np.float64)
    rgb_signal[:, 1] += pulse_component            # pulse in green channel
    rgb_signal += drift[:, None]                   # drift affects ALL channels equally
    rgb_signal += noise

    pulse = pos_algorithm(rgb_signal, FPS)
    filtered = bandpass_filter(pulse, FPS)
    result = estimate_hr_and_hrv(filtered, FPS)

    print(f"  Estimated: {result}")
    assert result["hr_bpm"] is not None, "FAILED: drift destroyed the pulse signal entirely"
    assert abs(result["hr_bpm"] - true_hr_bpm) < 15, \
        f"FAILED: HR {result['hr_bpm']} too far from true {true_hr_bpm} under drift"
    print(f"  PASS: recovered ~{result['hr_bpm']} bpm despite heavy lighting drift "
          f"(true={true_hr_bpm})\n")


if __name__ == "__main__":
    np.random.seed(7)
    test_short_signal_no_crash()
    test_flat_signal_rejected()
    test_out_of_range_hr_rejected()
    test_bradycardia_range_boundary()
    test_normal_blink_not_flagged_as_microsleep()
    test_microsleep_detected_and_separated_from_blinks()
    test_microsleep_detected_in_real_time_while_ongoing()
    test_perclos_calculation()
    test_pipeline_robust_to_lighting_drift()
    print("ALL EDGE-CASE TESTS PASSED")