"""Reading and writing landmark datasets. The same formats are used for synthetic data and for data you record yourself."""
import csv
from pathlib import Path

import numpy as np

HEADER = ["signer", "label", "left"] + [f"{a}{i}" for i in range(21) for a in "xyz"]


def write_static_csv(path, landmarks, labels, signers, left, append=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or not append
    with open(path, "a" if append else "w", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(HEADER)
        for lm, lab, sg, lf in zip(landmarks, labels, signers, left):
            w.writerow([sg, lab, int(lf)] + [f"{v:.5f}" for v in np.asarray(lm).ravel()])


def read_static_csv(path):
    rows = list(csv.reader(open(path)))[1:]
    signers = np.array([r[0] for r in rows]); labels = np.array([r[1] for r in rows]); left = np.array([bool(int(r[2])) for r in rows])
    lms = np.array([[float(v) for v in r[3:]] for r in rows]).reshape(-1, 21, 3)
    return lms, labels, signers, left


def write_dynamic_npz(path, seqs, labels, signers, left):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    T = max(len(s) for s in seqs); padded = np.full((len(seqs), T, 21, 3), np.nan)
    for i, s in enumerate(seqs): padded[i, :len(s)] = s
    np.savez_compressed(path, frames=padded, lengths=np.array([len(s) for s in seqs]), labels=np.array(labels), signers=np.array(signers), left=np.array(left))


def read_dynamic_npz(path):
    d = np.load(path, allow_pickle=False)
    frames, lengths = d["frames"], d["lengths"]                  # read each array ONCE (every d[...] access decompresses it again)
    seqs = [frames[i, :n] for i, n in enumerate(lengths)]
    return seqs, d["labels"], d["signers"], d["left"]
