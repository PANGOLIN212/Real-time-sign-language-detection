"""
Turn recorded videos into landmark rows. NOT TESTED in the build environment (needs the MediaPipe model and real videos of hands).

Expected layout:  videos/<signer>/<label>/*.mp4
Usage: python scripts/extract_from_videos.py --videos-dir videos --model-path hand_landmarker.task --every 2
Only landmarks are written (to data/video_landmarks.csv), not the video frames.
"""
import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hand_tracker import HandTracker
from src.io_utils import write_static_csv

ap = argparse.ArgumentParser()
ap.add_argument("--videos-dir", required=True); ap.add_argument("--model-path", default="hand_landmarker.task")
ap.add_argument("--every", type=int, default=2, help="keep every n-th frame"); ap.add_argument("--out", default=str(ROOT / "data" / "video_landmarks.csv"))
a = ap.parse_args()
try:
    tracker = HandTracker(a.model_path)
except FileNotFoundError as e:
    sys.exit(f"ERROR: {e}")
total = 0
for video in sorted(Path(a.videos_dir).glob("*/*/*.mp4")):
    signer, label = video.parent.parent.name, video.parent.name
    cap = cv2.VideoCapture(str(video)); n, kept = 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    while True:
        ok, frame = cap.read()
        if not ok: break
        if n % a.every == 0:
            hands = tracker.process(frame, int(n * 1000 / fps))
            if hands:
                write_static_csv(a.out, [hands[0]["landmarks"]], [label], [signer], [hands[0]["left"]], append=True); kept += 1
        n += 1
    cap.release(); total += kept
    print(f"{video}: {kept} frames with a hand out of {n}")
print(f"Wrote {total} rows to {a.out}")
