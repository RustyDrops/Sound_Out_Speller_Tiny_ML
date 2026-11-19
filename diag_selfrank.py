"""Diagnostic: which held-out-half words fail the correct-spelling self-rank,
and what beats them? Run after train_v1.py run 2."""
import json
import random
from pathlib import Path

import numpy as np

from features import pair_features
from train_v1 import LEVEL_IDX, LEVEL_ORDER, load_vocab

HERE = Path(__file__).parent
SEED = 42

words, lvl = load_vocab()
data = np.load(HERE / "model_v1.npz")
coef, intercept = data["coef"], float(data["intercept"][0])

id_words = words[:]
random.Random(SEED + 1).shuffle(id_words)
identity_train_words = set(id_words[: len(id_words) // 2])
held = [w for w in words if w not in identity_train_words]

fails = []
for w in held:
    feats = np.array([pair_features(w, c, lvl[c]) for c in words])
    scores = feats @ coef + intercept
    order = np.argsort(-scores, kind="stable")
    ranked = [words[i] for i in order]
    if ranked[0] != w:
        fails.append((w, ranked[:3]))
print(f"held-out half: {len(held)} words, {len(fails)} self-rank failures")
for w, top in fails:
    print(f"  attempt {w!r} -> top3 {top}")
