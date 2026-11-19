"""Sound-Out Speller v1 — runtime.

Pure NumPy. Loads model_v1.npz (logistic pair-ranker weights + vocabulary),
scores every vocabulary word for an attempt, and applies the deterministic
exact-match rule from SPEC.md: an attempt that exactly matches a vocabulary
word is trusted as correct and returned first.

Usage: python3 predict_v1.py sed becos wos said teh
"""
import sys
from pathlib import Path

import numpy as np

from features import pair_features

HERE = Path(__file__).parent
_data = np.load(HERE / "model_v1.npz", allow_pickle=False)
COEF = _data["coef"]
INTERCEPT = float(_data["intercept"][0])
VOCAB = [str(w) for w in _data["vocab"]]
LEVEL_IDX = {w: int(i) for w, i in zip(VOCAB, _data["level_idx"])}
VOCAB_SET = set(VOCAB)


def suggest(attempt: str, k: int = 5):
    a = attempt.lower().strip()
    feats = np.array([pair_features(a, c, LEVEL_IDX[c]) for c in VOCAB])
    scores = feats @ COEF + INTERCEPT
    order = np.argsort(-scores, kind="stable")
    ranked = [VOCAB[i] for i in order]
    exact = a in VOCAB_SET
    if exact:
        ranked.remove(a)
        ranked = [a] + ranked
    return ranked[:k], exact


if __name__ == "__main__":
    for att in sys.argv[1:]:
        top, exact = suggest(att)
        note = " (exact vocab match — trusted as correct)" if exact else ""
        print(f"{att!r} -> {top}{note}")
