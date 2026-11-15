"""
Real-time sign recognition demo.

  --tracker synthetic   replays a scripted stream of synthetic hands (no camera, no MediaPipe needed): TESTED
  --tracker mediapipe   uses your webcam or a video with MediaPipe HandLandmarker:  NOT TESTED in the build environment
                        (needs --model-path to hand_landmarker.task)

Examples
  python scripts/realtime_demo.py --tracker synthetic --headless --output demo.mp4
  python scripts/realtime_demo.py --tracker mediapipe --model-path hand_landmarker.task --source 0 --flip
Keys (window mode): q quit, c clear the text, r reset the recogniser.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import make_signers, sample_sequence, stream_from_schedule
from src.hand_model import DYNAMIC_SIGNS, STATIC_SIGNS
from src.hand_tracker import HandTracker, SyntheticTracker, draw_hand
from src.recognizer import DynamicRecognizer, StaticRecognizer

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="static", choices=["static", "dynamic"])
ap.add_argument("--tracker", default="synthetic", choices=["synthetic", "mediapipe"])
ap.add_argument("--source", default="0", help="camera index or video path (mediapipe tracker)")
ap.add_argument("--model-path", default="hand_landmarker.task"); ap.add_argument("--models-dir", default=str(ROOT / "models"))
ap.add_argument("--headless", action="store_true"); ap.add_argument("--output", default=None); ap.add_argument("--flip", action="store_true")
ap.add_argument("--max-frames", type=int, default=0); ap.add_argument("--conf", type=float, default=0.75)
ap.add_argument("--hold", type=int, default=8); ap.add_argument("--window", type=int, default=7); ap.add_argument("--seed", type=int, default=5)
a = ap.parse_args()

bundle = joblib.load(Path(a.models_dir) / f"{a.mode}_model.joblib")
rec = (StaticRecognizer(bundle["model"], bundle["classes"], a.window, a.conf, a.hold) if a.mode == "static"
       else DynamicRecognizer(bundle["model"], bundle["classes"], conf_thresh=min(a.conf, 0.7)))

expected, cap, left_flag = [], None, False
if a.tracker == "synthetic":
    rng = np.random.default_rng(a.seed); signer = make_signers(1, a.seed)[0]; left_flag = signer["left"]
    if a.mode == "static":
        signs = list(rng.choice(STATIC_SIGNS, 6, replace=False)); expected = [str(s) for s in signs]
        schedule = [(None, 12)]
        for s in signs: schedule += [(str(s), int(rng.integers(30, 46))), (None, 10)]
        frames, _ = stream_from_schedule(schedule, signer, seed=a.seed)
    else:
        signs = list(rng.choice(DYNAMIC_SIGNS, 4, replace=False)); expected = [str(s) for s in signs]
        frames = [None] * 10
        for s in signs: frames += list(sample_sequence(rng, signer, str(s), drop_p=0.0)) + [None] * 12
    tracker = SyntheticTracker(frames, left_flag); n_frames = len(frames)
else:
    try:
        tracker = HandTracker(a.model_path)
    except FileNotFoundError as e:
        sys.exit(f"ERROR: {e}")
    cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    if not cap.isOpened(): sys.exit(f"ERROR: could not open video source {a.source}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not a.source.isdigit() else 0

W, H, FPS = 640, 480, 30
writer = cv2.VideoWriter(a.output, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H)) if a.output else None
t_start, i, commits, t_track, t_rec = time.time(), 0, [], 0.0, 0.0
last_info = {"label": None, "confidence": 0.0}
while True:
    if cap is not None:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.resize(frame, (W, H))
        if a.flip: frame = cv2.flip(frame, 1)
    else:
        if i >= n_frames: break
        frame = np.full((H, W, 3), 30, np.uint8)
    t0 = time.time(); hands = tracker.process(frame, i * 1000 // FPS); t1 = time.time()
    hand = hands[0] if hands else None
    info = rec.update(hand["landmarks"] if hand else None, hand["left"] if hand else False); t2 = time.time()
    t_track += t1 - t0; t_rec += t2 - t1
    if info.get("commit"): commits.append((i, info["commit"])); print(f"frame {i:4d}: committed '{info['commit']}'")
    if hand: draw_hand(frame, hand["landmarks"])
    label = info.get("label") or ("(no hand)" if not hand else "...")
    cv2.putText(frame, f"{label}  {info.get('confidence', 0):.2f}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.rectangle(frame, (15, 50), (15 + int(200 * info.get("progress", 0)), 62), (0, 200, 0), -1)
    cv2.putText(frame, "Text: " + " ".join(rec.text[-6:]), (15, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"{(i + 1) / max(time.time() - t_start, 1e-6):.0f} fps", (W - 110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    if writer: writer.write(frame)
    if not a.headless:
        cv2.imshow("Sign recognition (q to quit)", frame); k = cv2.waitKey(1) & 0xFF
        if k == ord("q"): break
        if k == ord("c"): rec.text.clear()
        if k == ord("r"): rec.reset()
    i += 1
    if a.max_frames and i >= a.max_frames: break
if cap is not None: cap.release()
if writer: writer.release()
cv2.destroyAllWindows() if not a.headless else None
print(f"\nProcessed {i} frames. Tracker {1000 * t_track / max(i, 1):.2f} ms/frame, recogniser {1000 * t_rec / max(i, 1):.2f} ms/frame")
print("Committed:", [c for _, c in commits])
if expected: print("Expected :", expected)
