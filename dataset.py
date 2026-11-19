"""Synthetic landmark datasets built on src/hand_model.py: many 'signers', static handshapes and dynamic motions."""
import numpy as np

from .hand_model import (DYNAMIC_SIGNS, POSES, STATIC_SIGNS, hand_points, interpolate_pose, make_signer, motion, project)


def make_signers(n, seed=0):
    rng = np.random.default_rng(seed)
    return [make_signer(rng) for _ in range(n)]


def sample_static(rng, signer, sign, tilt=20.0, roll=35.0, jitter_scale=1.0, angle_noise=9.0, pose=None):
    """One frame of a static sign as MediaPipe-style landmarks (21, 3)."""
    pts = hand_points(pose or POSES[sign], signer, rng, angle_noise, name=None if pose else sign)
    rot = (rng.uniform(-tilt, tilt), rng.uniform(-tilt, tilt), rng.uniform(-roll, roll))
    size = signer["size"] * rng.uniform(0.85, 1.15)
    centre = (rng.uniform(0.25, 0.75), rng.uniform(0.3, 0.7))
    return project(pts, size, centre, rot, signer["left"], signer["jitter"] * jitter_scale, rng)


def make_static_dataset(signers, per_class=60, tilt=20.0, jitter_scale=1.0, seed=0, signs=STATIC_SIGNS):
    """Returns landmarks (N, 21, 3), labels (N,), signer ids (N,), left-hand flags (N,)."""
    rng = np.random.default_rng(seed)
    L, y, g, left = [], [], [], []
    for gi, sg in enumerate(signers):
        for sign in signs:
            for _ in range(per_class):
                L.append(sample_static(rng, sg, sign, tilt, 35.0, jitter_scale)); y.append(sign); g.append(gi); left.append(sg["left"])
    return np.array(L), np.array(y), np.array(g), np.array(left)


def sample_sequence(rng, signer, sign, tilt=15.0, jitter_scale=1.0, drop_p=0.03, angle_noise=8.0):
    """One dynamic sign: (T, 21, 3) landmarks, with NaN rows where the tracker lost the hand."""
    T = int(rng.integers(24, 41))
    t = np.linspace(0, 1, T); warp = rng.uniform(-0.1, 0.1); t = np.clip(t + warp * np.sin(np.pi * t), 0, 1)
    centre = np.array([rng.uniform(0.35, 0.65), rng.uniform(0.35, 0.55)])
    size0 = signer["size"] * rng.uniform(0.85, 1.15)
    rx, ry, rz = rng.uniform(-tilt, tilt), rng.uniform(-tilt, tilt), rng.uniform(-20, 20)
    mirror = -1.0 if signer["left"] else 1.0
    amp = rng.uniform(0.45, 1.5)                                  # how large and energetic this person makes the movement
    frames = []
    for ti in t:
        pose, dx, dy, sf, roll = motion(sign, ti)
        dx, dy, sf, roll = amp * dx, amp * dy, 1 + amp * (sf - 1), amp * roll
        pts = hand_points(pose, signer, rng, angle_noise)
        frames.append(project(pts, size0 * sf, centre + np.array([mirror * dx, dy]), (rx, ry, rz + roll), signer["left"], signer["jitter"] * jitter_scale, rng))
    frames = np.array(frames)
    drop = rng.random(T) < drop_p
    for i in np.nonzero(drop)[0]:                               # occasional bursts of lost frames
        frames[i:i + int(rng.integers(1, 3))] = np.nan
    return frames


def make_dynamic_dataset(signers, per_class=15, tilt=15.0, jitter_scale=1.0, drop_p=0.03, seed=0):
    """Returns a list of (T, 21, 3) arrays, labels, signer ids, left flags."""
    rng = np.random.default_rng(seed)
    seqs, y, g, left = [], [], [], []
    for gi, sg in enumerate(signers):
        for sign in DYNAMIC_SIGNS:
            for _ in range(per_class):
                seqs.append(sample_sequence(rng, sg, sign, tilt, jitter_scale, drop_p)); y.append(sign); g.append(gi); left.append(sg["left"])
    return seqs, np.array(y), np.array(g), np.array(left)


def stream_from_schedule(schedule, signer, seed=0, transition=6, absent_gap=8, tilt=15.0):
    """
    A continuous landmark stream for the real-time tests. schedule = [(sign or None, n_frames), ...] where None means no hand.
    Between two signs the pose blends over `transition` frames. Returns (frames list with None for no hand, truth label per frame).
    """
    rng = np.random.default_rng(seed)
    frames, truth = [], []
    prev = None
    rot = (rng.uniform(-tilt, tilt), rng.uniform(-tilt, tilt), rng.uniform(-15, 15))
    size = signer["size"]; centre = (rng.uniform(0.4, 0.6), rng.uniform(0.4, 0.6))
    for sign, n in schedule:
        if sign is None:
            frames += [None] * n; truth += [None] * n; prev = None; continue
        for i in range(n):
            if prev is not None and i < transition:
                pose = interpolate_pose(POSES[prev], POSES[sign], (i + 1) / (transition + 1)); lab = None
            else:
                pose = POSES[sign]; lab = sign
            pts = hand_points(pose, signer, rng, 4.0)
            frames.append(project(pts, size, (centre[0] + rng.normal(0, 0.003), centre[1] + rng.normal(0, 0.003)), rot, signer["left"], signer["jitter"], rng)); truth.append(lab)
        prev = sign
    return frames, truth
