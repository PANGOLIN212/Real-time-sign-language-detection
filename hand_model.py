"""
A small forward-kinematics model of a hand that produces the same 21 landmarks as MediaPipe Hands.

Landmark order (MediaPipe):  0 wrist | 1-4 thumb (CMC, MCP, IP, TIP) | 5-8 index (MCP, PIP, DIP, TIP)
                             9-12 middle | 13-16 ring | 17-20 pinky

Why this exists: the pipeline (features, classifiers, real-time logic) can be built and tested without a camera or
a real dataset. The handshapes below are GENERIC shapes with descriptive names. They are NOT Kenyan Sign Language
signs, and real accuracy can only be measured on landmarks recorded from real signers.

Coordinates in the hand frame: wrist at the origin, the middle-finger knuckle (MCP) at (0, 1, 0), fingers point to +y,
the palm faces -z. Output landmarks are in normalised image coordinates (x, y in [0, 1], y downwards, z relative).
"""
import numpy as np

STATIC_SIGNS = ["open_palm", "fist", "point", "peace", "thumbs_up", "ok_ring", "pinky_up", "three", "claw", "flat_thumb_in"]
DYNAMIC_SIGNS = ["wave", "circle", "circle_ccw", "push", "swipe_down", "swipe_up", "open_close", "tap"]

FINGERS = ["index", "middle", "ring", "pinky"]
BASE = {"index": (-0.34, 0.95), "middle": (0.0, 1.0), "ring": (0.30, 0.94), "pinky": (0.58, 0.82)}
SEG = {"index": (0.45, 0.28, 0.25), "middle": (0.50, 0.31, 0.27), "ring": (0.46, 0.29, 0.26), "pinky": (0.36, 0.22, 0.22)}
SPREAD = {"index": -8.0, "middle": 0.0, "ring": 6.0, "pinky": 14.0}          # natural fan-out of the fingers, degrees
THUMB_CMC, THUMB_SEG = (-0.32, 0.22), (0.42, 0.32, 0.28)

EXT, CURL = (0, 0, 0), (0, 85, 100)                                          # (abduction, MCP flex, PIP flex), degrees
POSES = {   # finger: (abduction offset, MCP flexion, PIP flexion);  thumb: (abduction, flexion 1, flexion 2)
    "open_palm":     {"index": EXT, "middle": EXT, "ring": EXT, "pinky": EXT, "thumb": (25, 0, 0), "spread": 1.0},
    "fist":          {"index": CURL, "middle": CURL, "ring": CURL, "pinky": CURL, "thumb": (-10, 35, 40), "spread": 0.0},
    "point":         {"index": EXT, "middle": CURL, "ring": CURL, "pinky": CURL, "thumb": (-5, 30, 35), "spread": 0.3},
    "peace":         {"index": (-10, 0, 0), "middle": (8, 0, 0), "ring": CURL, "pinky": CURL, "thumb": (-5, 30, 35), "spread": 0.5},
    "thumbs_up":     {"index": CURL, "middle": CURL, "ring": CURL, "pinky": CURL, "thumb": (25, 0, 0), "spread": 0.0},
    "ok_ring":       {"index": (0, 40, 55), "middle": EXT, "ring": EXT, "pinky": EXT, "thumb": (10, 30, 25), "spread": 0.8},
    "pinky_up":      {"index": CURL, "middle": CURL, "ring": CURL, "pinky": EXT, "thumb": (-5, 30, 35), "spread": 0.3},
    "three":         {"index": EXT, "middle": EXT, "ring": EXT, "pinky": CURL, "thumb": (-5, 30, 35), "spread": 0.6},
    "claw":          {"index": (0, 35, 55), "middle": (0, 35, 55), "ring": (0, 35, 55), "pinky": (0, 35, 55), "thumb": (10, 20, 25), "spread": 0.6},
    "flat_thumb_in": {"index": EXT, "middle": EXT, "ring": EXT, "pinky": EXT, "thumb": (-15, 40, 35), "spread": 0.0},
}


def _finger(base, seg, abduct, mcp, pip, length):
    p = np.array([base[0], base[1], 0.0]); pts = [p.copy()]
    a = np.deg2rad(abduct); u = np.array([np.sin(a), np.cos(a), 0.0]); cum = 0.0
    for L, f in zip(seg, (mcp, pip, 0.7 * pip)):
        cum += np.deg2rad(f)
        d = u * np.cos(cum) + np.array([0.0, 0.0, -1.0]) * np.sin(cum)
        p = p + d * L * length; pts.append(p.copy())
    return pts


def _thumb(abduct, f1, f2, scale):
    p = np.array([THUMB_CMC[0], THUMB_CMC[1], 0.0]); pts = [p.copy()]
    ang = np.deg2rad(40 + abduct); u = np.array([-np.sin(ang), np.cos(ang), 0.0])
    w = np.array([0.7, 0.2, -0.7]); w /= np.linalg.norm(w); cum = 0.0
    for L, f in zip(THUMB_SEG, (f1, f2, 0.8 * f2)):
        cum += np.deg2rad(f)
        d = u * np.cos(cum) + w * np.sin(cum); p = p + d * L * scale; pts.append(p.copy())
    return pts


def hand_points(pose, signer=None, rng=None, angle_noise=0.0, name=None):
    """Landmarks (21, 3) in the hand frame for a pose dict, with optional signer traits and per-sample noise.
    If `name` is given, the signer's personal way of making that sign (a fixed offset per joint) is applied."""
    rng = rng or np.random.default_rng(0)
    gain = signer["flex_gain"] if signer else 1.0
    lens = signer["finger_len"] if signer else {k: 1.0 for k in FINGERS + ["thumb"]}
    spread_off = signer["spread_off"] if signer else 0.0
    n = lambda s: rng.normal(0, s) if angle_noise else 0.0
    quirk = signer["quirks"].get(name) if (signer and name) else None
    if quirk is not None:
        pose = {**pose, "thumb": tuple(np.array(pose["thumb"]) + quirk["thumb"]), **{f: tuple(np.array(pose[f]) + quirk[f]) for f in FINGERS}}
    pts = [np.zeros(3)]
    t = pose["thumb"]
    pts += _thumb(t[0] + n(angle_noise / 2), max(t[1] * gain + n(angle_noise), 0), max(t[2] * gain + n(angle_noise), 0), lens["thumb"])
    for f in FINGERS:
        ab, mcp, pip = pose[f]
        ab = ab + SPREAD[f] * (pose["spread"] + spread_off) + n(angle_noise / 2)
        pts += _finger(BASE[f], SEG[f], ab, max(mcp * gain + n(angle_noise), 0), max(pip * gain + n(angle_noise), 0), lens[f])
    return np.array(pts)


def interpolate_pose(a, b, w):
    """Blend two pose dicts (w = 0 gives a, w = 1 gives b). Used for transitions and open/close motions."""
    out = {"spread": a["spread"] * (1 - w) + b["spread"] * w, "thumb": tuple(np.array(a["thumb"]) * (1 - w) + np.array(b["thumb"]) * w)}
    for f in FINGERS:
        out[f] = tuple(np.array(a[f]) * (1 - w) + np.array(b[f]) * w)
    return out


def rotation(rx, ry, rz):
    rx, ry, rz = np.deg2rad([rx, ry, rz])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def project(points, size, centre, rot=(0, 0, 0), left=False, jitter=0.0, rng=None):
    """Hand-frame points to normalised image coordinates: rotate, mirror for a left hand, scale, translate, add jitter."""
    rng = rng or np.random.default_rng(0)
    p = points @ rotation(*rot).T
    if left:
        p = p * np.array([-1, 1, 1])
    out = np.empty_like(p)
    out[:, 0] = centre[0] + size * p[:, 0]
    out[:, 1] = centre[1] - size * p[:, 1]                 # image y points down
    out[:, 2] = size * p[:, 2]
    if jitter:
        out += rng.normal(0, jitter, out.shape) * np.array([1.0, 1.0, 3.0])      # MediaPipe's depth (z) is much noisier than x and y
    return out


def make_signer(rng):
    """A signer with a hand size, finger proportions, flexibility, handedness, tracker jitter, and a PERSONAL way of making each sign."""
    quirks = {name: {**{f: rng.normal(0, 9, 3) for f in FINGERS}, "thumb": rng.normal(0, 12, 3)} for name in POSES}
    return {"flex_gain": rng.uniform(0.85, 1.15), "spread_off": rng.uniform(-0.2, 0.2),
            "finger_len": {k: rng.uniform(0.9, 1.1) for k in FINGERS + ["thumb"]},
            "size": rng.uniform(0.11, 0.20), "left": bool(rng.random() < 0.15), "jitter": rng.uniform(0.003, 0.010), "quirks": quirks}


# --------------------------------------------------------------------------------------- dynamic signs
def motion(sign, t):
    """For a dynamic sign at normalised time t in [0, 1]: (pose, dx, dy, size_factor, roll_degrees). dx, dy in image units."""
    a = lambda name: POSES[name]
    if sign == "wave":
        return a("open_palm"), 0.12 * np.sin(2 * np.pi * 2 * t), 0.0, 1.0, 18 * np.sin(2 * np.pi * 2 * t)
    if sign == "circle":
        return a("point"), 0.09 * np.cos(2 * np.pi * t) - 0.09, 0.09 * np.sin(2 * np.pi * t), 1.0, 0.0
    if sign == "push":
        return a("flat_thumb_in"), 0.0, 0.0, 1.0 + 0.45 * t, 0.0
    if sign == "circle_ccw":
        return a("point"), 0.09 * np.cos(-2 * np.pi * t) - 0.09, 0.09 * np.sin(-2 * np.pi * t), 1.0, 0.0
    if sign == "swipe_down":
        return a("peace"), 0.0, 0.30 * t, 1.0, 0.0
    if sign == "swipe_up":
        return a("peace"), 0.0, -0.30 * t, 1.0, 0.0
    if sign == "open_close":
        w = 0.5 * (1 - np.cos(2 * np.pi * t))
        return interpolate_pose(a("open_palm"), a("fist"), w), 0.0, 0.0, 1.0, 0.0
    if sign == "tap":
        return a("point"), 0.0, 0.06 * np.abs(np.sin(2 * np.pi * 2 * t)), 1.0, 0.0
    raise ValueError(sign)
