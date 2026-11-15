"""Create the synthetic landmark datasets: data/static_landmarks.csv and data/dynamic_sequences.npz.  Usage: python scripts/generate_synthetic_data.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import make_dynamic_dataset, make_signers, make_static_dataset
from src.io_utils import write_dynamic_npz, write_static_csv

DATA = Path(__file__).resolve().parents[1] / "data"
signers = make_signers(12, seed=1)
L, y, g, left = make_static_dataset(signers, per_class=60, seed=2)
write_static_csv(DATA / "static_landmarks.csv", L, y, [f"s{i:02d}" for i in g], left)
seqs, yd, gd, ld = make_dynamic_dataset(signers, per_class=15, seed=3)
write_dynamic_npz(DATA / "dynamic_sequences.npz", seqs, yd, [f"s{i:02d}" for i in gd], ld)
print(f"static: {len(y):,} frames from {len(signers)} signers | dynamic: {len(seqs):,} sequences | saved in {DATA}")
