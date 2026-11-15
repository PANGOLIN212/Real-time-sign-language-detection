"""
Record labelled landmarks from a webcam. NOT TESTED in the build environment (needs a camera and the MediaPipe model).

Static sign : python scripts/collect_data.py --label A --signer s01 --model-path hand_landmarker.task
              SPACE starts and stops recording; frames with a detected hand are appended to data/my_static_landmarks.csv
Dynamic sign: python scripts/collect_data.py --mode dynamic --label wave --signer s01 ...
              press R to start and stop recording ONE sequence; each is saved as data/my_sequences/<label>_<signer>_<n>.npz

ONLY landmarks are saved (21 points per frame), never video, which also protects the signer's privacy.
Get informed consent from every signer, and involve Deaf community members and KSL signers in deciding the vocabulary.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hand_tracker import HandTracker, draw_hand
from src.io_utils import write_dynamic_npz, write_static_csv

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True); ap.add_argument("--signer", required=True); ap.add_argument("--mode", default="static", choices=["static", "dynamic"])
ap.add_argument("--model-path", default="hand_landmarker.task"); ap.add_argument("--source", default="0"); ap.add_argument("--target", type=int, default=200)
ap.add_argument("--out", default=str(ROOT / "data" / "my_static_landmarks.csv")); ap.add_argument("--flip", action="store_true")
a = ap.parse_args()
try:
    tracker = HandTracker(a.model_path)
except FileNotFoundError as e:
    sys.exit(f"ERROR: {e}")
cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
if not cap.isOpened(): sys.exit(f"ERROR: could not open {a.source}")
recording, saved, seq, n_seq, t0 = False, 0, [], 0, time.time()
seq_dir = ROOT / "data" / "my_sequences"
while True:
    ok, frame = cap.read()
    if not ok: break
    if a.flip: frame = cv2.flip(frame, 1)
    hands = tracker.process(frame, int((time.time() - t0) * 1000)); hand = hands[0] if hands else None
    if hand: draw_hand(frame, hand["landmarks"])
    if recording:
        if a.mode == "static" and hand:
            write_static_csv(a.out, [hand["landmarks"]], [a.label], [a.signer], [hand["left"]], append=True); saved += 1
            if saved >= a.target: recording = False; print(f"Reached {saved} frames for '{a.label}'")
        elif a.mode == "dynamic":
            seq.append(hand["landmarks"] if hand else np.full((21, 3), np.nan)); left = hand["left"] if hand else False
    status = f"{'REC' if recording else 'idle'} {a.label} | saved {saved if a.mode == 'static' else n_seq}"
    cv2.putText(frame, status, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255) if recording else (255, 255, 255), 2)
    cv2.imshow("Collect landmarks (q quits)", frame); k = cv2.waitKey(1) & 0xFF
    if k == ord("q"): break
    if a.mode == "static" and k == ord(" "): recording = not recording
    if a.mode == "dynamic" and k == ord("r"):
        if recording and len(seq) > 8:
            n_seq += 1; write_dynamic_npz(seq_dir / f"{a.label}_{a.signer}_{n_seq}.npz", [np.array(seq)], [a.label], [a.signer], [left]); print(f"Saved sequence {n_seq}")
        seq, recording = [], not recording
cap.release(); cv2.destroyAllWindows()
