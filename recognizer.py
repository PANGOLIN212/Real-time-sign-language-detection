"""
Real-time recognition logic that sits between a hand tracker and the user interface.

A classifier gives noisy frame-by-frame guesses. A usable system must
  * smooth them over a few frames,
  * ignore low-confidence guesses,
  * commit a sign only after it has been held steadily (so passing handshapes are not typed),
  * avoid repeating the same sign while it is still held, and reset when the hand leaves the frame.
These functions are independent of MediaPipe and OpenCV, so they can be tested with recorded or synthetic landmark streams.
"""
from collections import deque

import numpy as np

from .features import dynamic_features, static_features


class StaticRecognizer:
    def __init__(self, model, classes, window=7, conf_thresh=0.75, hold_frames=8, absent_frames=6):
        self.model, self.classes = model, np.asarray(classes)
        self.window, self.conf_thresh, self.hold, self.absent_frames = window, conf_thresh, hold_frames, absent_frames
        self.reset()

    def reset(self):
        self.buf = deque(maxlen=self.window); self.current = None; self.run = 0; self.committed = None; self.absent = 0
        self.text = []; self.last = {"label": None, "confidence": 0.0}

    def update(self, landmarks=None, left=False):
        """Feed one frame (landmarks (21,3) or None when no hand was found). Returns a dict describing the current state."""
        if landmarks is None:
            self.absent += 1
            if self.absent >= self.absent_frames:                    # the hand really left: forget everything about the current sign
                self.buf.clear(); self.current = None; self.run = 0; self.committed = None
            return {"hand_present": False, "label": None, "confidence": 0.0, "commit": None, "progress": 0.0}
        self.absent = 0
        self.buf.append(self.model.predict_proba(static_features(landmarks, left)[None])[0])
        mean = np.mean(self.buf, axis=0); i = int(np.argmax(mean)); conf = float(mean[i])
        label = str(self.classes[i]) if conf >= self.conf_thresh else None
        if label is not None and label == self.current:
            self.run += 1
        else:
            self.current, self.run = label, (1 if label else 0)
        commit = None
        if label is not None and self.run >= self.hold and label != self.committed:
            commit = label; self.committed = label; self.text.append(label)
        self.last = {"label": label, "confidence": conf}
        return {"hand_present": True, "label": label, "confidence": conf, "commit": commit, "progress": min(self.run / self.hold, 1.0) if label else 0.0}


class DynamicRecognizer:
    """
    Segment-based recogniser for signs that involve motion. It collects frames while a hand is visible, and classifies the WHOLE
    movement once the hand has been gone for `end_gap` frames (a hand enters, signs, and leaves). Short tracker dropouts inside a
    sign are kept as gaps and interpolated by the features.

    Why not a sliding window? A window sees only part of a movement, which looks different from the complete movements the model
    was trained on, so it misses signs. Continuous signing without the hand leaving needs a start/end detector (for example from
    hand speed), which is a good extension.
    """
    def __init__(self, model, classes, min_len=12, end_gap=5, conf_thresh=0.6, max_len=90):
        self.model, self.classes = model, np.asarray(classes)
        self.min_len, self.end_gap, self.conf_thresh, self.max_len = min_len, end_gap, conf_thresh, max_len
        self.reset()

    def reset(self):
        self.seg, self.gap, self.text = [], 0, []

    def _finish(self, left):
        while self.seg and np.isnan(self.seg[-1][0, 0]):
            self.seg.pop()
        frames, self.seg, self.gap = np.array(self.seg), [], 0
        if len(frames) < self.min_len:
            return {"label": None, "confidence": 0.0, "commit": None}
        feats = dynamic_features(frames, left)
        if feats is None:
            return {"label": None, "confidence": 0.0, "commit": None}
        proba = self.model.predict_proba(feats[None])[0]; i = int(np.argmax(proba)); conf = float(proba[i])
        label = str(self.classes[i]) if conf >= self.conf_thresh else None
        if label is not None:
            self.text.append(label)
        return {"label": label, "confidence": conf, "commit": label}

    def update(self, landmarks=None, left=False):
        out = {"hand_present": landmarks is not None, "label": None, "confidence": 0.0, "commit": None, "progress": min(len(self.seg) / 30, 1.0)}
        if landmarks is not None:
            self.seg.append(np.asarray(landmarks, float)); self.gap = 0
            if len(self.seg) >= self.max_len:
                out.update(self._finish(left))
        elif self.seg:
            self.seg.append(np.full((21, 3), np.nan)); self.gap += 1
            if self.gap >= self.end_gap:
                out.update(self._finish(left))
        return out
