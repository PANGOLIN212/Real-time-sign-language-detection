"""
Train the static (handshape) and dynamic (motion) sign models and save them to models/.
Uses data/static_landmarks.csv and data/dynamic_sequences.npz (synthetic, or your own recordings in the same format).
Reports signer-independent accuracy: the models are always tested on signers they have never seen.

Usage: python scripts/train.py [--static-model svc|rf|lr|hgb] [--dynamic-model rf|hgb|lr]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import dynamic_features, static_features
from src.io_utils import read_dynamic_npz, read_static_csv

MODELS = {
    "svc": lambda: make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=True, random_state=0)),
    "lr": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)),
    "rf": lambda: RandomForestClassifier(300, random_state=0, n_jobs=1),
    "hgb": lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0),
}

ap = argparse.ArgumentParser()
ap.add_argument("--static-model", default="svc", choices=MODELS)
ap.add_argument("--dynamic-model", default="rf", choices=MODELS)
ap.add_argument("--data-dir", default=str(ROOT / "data")); ap.add_argument("--out-dir", default=str(ROOT / "models"))
a = ap.parse_args()
data, out = Path(a.data_dir), Path(a.out_dir); out.mkdir(exist_ok=True)
if not (data / "static_landmarks.csv").exists():
    subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_synthetic_data.py")], check=True)

metrics = {}
lms, y, sg, left = read_static_csv(data / "static_landmarks.csv")
X = np.array([static_features(l, lf) for l, lf in zip(lms, left)])
accs = []
for tr, te in GroupKFold(4).split(X, y, sg):
    accs.append(MODELS[a.static_model]().fit(X[tr], y[tr]).score(X[te], y[te]))
print(f"Static  ({a.static_model}): signer-independent accuracy {np.mean(accs):.3f} (folds {np.round(accs, 3)})")
clf = MODELS[a.static_model]().fit(X, y)
joblib.dump({"model": clf, "classes": list(clf.classes_), "kind": "static"}, out / "static_model.joblib")
metrics["static"] = {"model": a.static_model, "cv_accuracy": float(np.mean(accs)), "classes": list(clf.classes_)}

if (data / "dynamic_sequences.npz").exists():
    seqs, yd, sd, ld = read_dynamic_npz(data / "dynamic_sequences.npz")
    feats = [dynamic_features(s, lf) for s, lf in zip(seqs, ld)]
    keep = np.array([f is not None for f in feats]); Xd = np.array([f for f in feats if f is not None]); yd, sd = yd[keep], sd[keep]
    accs = [MODELS[a.dynamic_model]().fit(Xd[tr], yd[tr]).score(Xd[te], yd[te]) for tr, te in GroupKFold(4).split(Xd, yd, sd)]
    print(f"Dynamic ({a.dynamic_model}): signer-independent accuracy {np.mean(accs):.3f} (folds {np.round(accs, 3)})")
    clf = MODELS[a.dynamic_model]().fit(Xd, yd)
    joblib.dump({"model": clf, "classes": list(clf.classes_), "kind": "dynamic"}, out / "dynamic_model.joblib")
    metrics["dynamic"] = {"model": a.dynamic_model, "cv_accuracy": float(np.mean(accs)), "classes": list(clf.classes_)}
json.dump(metrics, open(out / "metrics.json", "w"), indent=1)
print(f"Saved models to {out}")
