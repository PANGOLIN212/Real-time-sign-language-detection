"""
Landmark features for sign recognition.

normalize(): makes a hand invariant to WHERE it is in the frame, HOW BIG it is, which way it is ROTATED in the image plane,
and whether it is a LEFT or RIGHT hand. After this, only the handshape is left.
static_features(): 87 numbers per frame (normalised coordinates, joint angles, fingertip distances).
dynamic_features(): a fixed-length vector for a whole sequence (handshape over time + where the hand moved).
"""
import numpy as np

TIPS = [4, 8, 12, 16, 20]
CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
N_STATIC = 87
N_STEPS = 16
N_PER_FRAME = 24
N_DYNAMIC = N_STEPS * N_PER_FRAME + 7


def normalize(lm, left=False):
    """Centre on the wrist, mirror left hands, scale so wrist-to-middle-knuckle = 1, rotate so the fingers point up. Returns (points, original scale)."""
    p = np.array(lm, dtype=float) - np.asarray(lm)[0]
    if left:
        p[:, 0] *= -1
    scale = np.linalg.norm(p[9]) + 1e-9
    p /= scale
    delta = -np.pi / 2 - np.arctan2(p[9, 1], p[9, 0])
    c, s = np.cos(delta), np.sin(delta)
    x, y = p[:, 0].copy(), p[:, 1].copy()
    p[:, 0], p[:, 1] = c * x - s * y, s * x + c * y
    return p, scale


def joint_angles(p):
    """15 joint angles (radians): 3 per finger, the bend between consecutive bones."""
    out = []
    for chain in CHAINS:
        bones = [p[chain[i + 1]] - p[chain[i]] for i in range(4)]
        for i in range(3):
            a, b = bones[i], bones[i + 1]
            out.append(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9), -1, 1)))
    return np.array(out)


def static_features(lm, left=False):
    p, _ = normalize(lm, left)
    ang = joint_angles(p)
    dist = [np.linalg.norm(p[4] - p[i]) for i in TIPS[1:]]                                  # thumb tip to each other tip
    dist += [np.linalg.norm(p[TIPS[i]] - p[TIPS[i + 1]]) for i in range(1, 4)]              # neighbouring fingertips
    dist += [np.linalg.norm(p[i]) for i in TIPS]                                            # tip to wrist (finger extension)
    return np.concatenate([p[1:].ravel(), ang, dist])


def raw_features(lm):
    """Baseline for ablations: the raw image coordinates with no normalisation."""
    return np.asarray(lm, float).ravel()


def interpolate_nans(frames):
    f = np.array(frames, dtype=float)
    valid = ~np.isnan(f[:, 0, 0])
    if valid.sum() < 4:
        return None
    idx = np.arange(len(f))
    for j in range(21):
        for c in range(3):
            f[:, j, c] = np.interp(idx, idx[valid], f[valid, j, c])
    return f


def _wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi


def dynamic_features(frames, left=False):
    """frames: (T, 21, 3) with NaN rows for lost frames. Returns a vector of length N_DYNAMIC, or None if too few frames."""
    f = interpolate_nans(frames)
    if f is None:
        return None
    if left:
        f[:, :, 0] *= -1
    T = len(f)
    s0 = np.linalg.norm(f[0, 9] - f[0, 0]) + 1e-9
    w0 = f[0, 0, :2].copy()
    v0 = f[0, 9, :2] - f[0, 0, :2]; o0 = np.arctan2(v0[1], v0[0])
    rows = []
    for t in range(T):
        p, sc = normalize(f[t])
        v = f[t, 9, :2] - f[t, 0, :2]
        rows.append(np.concatenate([joint_angles(p), [np.linalg.norm(p[i]) for i in TIPS], (f[t, 0, :2] - w0) / s0,
                                    [np.log(sc / s0)], [_wrap(np.arctan2(v[1], v[0]) - o0)]]))
    pf = np.array(rows)
    grid = np.linspace(0, T - 1, N_STEPS)
    res = np.stack([np.interp(grid, np.arange(T), pf[:, c]) for c in range(pf.shape[1])], axis=1)
    path = np.linalg.norm(np.diff(pf[:, 20:22], axis=0), axis=1)
    extras = [path.sum(), np.linalg.norm(pf[-1, 20:22]), pf[:, 20].std(), pf[:, 21].std(), np.ptp(pf[:, 22]), path.mean(), np.ptp(pf[:, 23])]
    return np.concatenate([res.ravel(), extras])


def augment_tilt(lm, rng, max_tilt=30.0):
    """3D rotation augmentation about the wrist, mimicking a hand turned toward or away from the camera."""
    from .hand_model import rotation
    R = rotation(rng.uniform(-max_tilt, max_tilt), rng.uniform(-max_tilt, max_tilt), 0.0)
    c = lm - lm[0]
    return c @ R.T + lm[0]
