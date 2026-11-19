"""
Hand trackers and drawing helpers.

HandTracker wraps MediaPipe's HandLandmarker (the Tasks API, which replaced the old `mp.solutions.hands`).
*** HandTracker is NOT TESTED here: it needs a downloaded model file and a real camera or video. ***
SyntheticTracker replays a scripted landmark stream so the rest of the application can be run and tested anywhere.
"""
from pathlib import Path

import numpy as np

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
               (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
MODEL_HELP = ("MediaPipe needs the 'hand_landmarker.task' model file. Download it from the MediaPipe Hand Landmarker documentation "
              "(https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) and pass its path with --model-path.")


class HandTracker:
    def __init__(self, model_path, num_hands=1, min_conf=0.5):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}\n{MODEL_HELP}")
        import mediapipe as mp                                        # imported here so the rest of the project works without it
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self.mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)), running_mode=vision.RunningMode.VIDEO, num_hands=num_hands,
            min_hand_detection_confidence=min_conf, min_hand_presence_confidence=min_conf, min_tracking_confidence=min_conf)
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process(self, frame_bgr, timestamp_ms):
        """Returns a list of dicts: {"landmarks": (21,3) array in normalised image coordinates, "left": bool, "score": float}."""
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.detector.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), int(timestamp_ms))
        hands = []
        for lms, handed in zip(result.hand_landmarks, result.handedness):
            # MediaPipe assumes a mirrored (selfie) image. If you do not flip the frame, its Left/Right labels are swapped: check on your setup.
            hands.append({"landmarks": np.array([[p.x, p.y, p.z] for p in lms]), "left": handed[0].category_name == "Left", "score": handed[0].score})
        return hands


class SyntheticTracker:
    """Replays a precomputed landmark stream (from src.dataset.stream_from_schedule) as if it came from a camera."""
    def __init__(self, frames, left=False):
        self.frames, self.left, self.i = frames, left, 0

    def process(self, frame_bgr=None, timestamp_ms=None):
        lm = self.frames[self.i] if self.i < len(self.frames) else None
        self.i += 1
        return [] if lm is None else [{"landmarks": lm, "left": self.left, "score": 1.0}]


def draw_hand(frame, landmarks, colour=(0, 200, 0)):
    import cv2
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], colour, 2)
    for p in pts:
        cv2.circle(frame, p, 3, (0, 0, 255), -1)
