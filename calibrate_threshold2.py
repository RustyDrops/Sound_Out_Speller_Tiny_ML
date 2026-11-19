"""Calibrate v2 of the demo confidence rule (calibrate_threshold.py showed the
raw logistic probability does not separate right from wrong top-1 picks).

The demo's "not in this level's words yet" message exists to catch
OUT-OF-VOCAB attempts (e.g. 'fone' for phone, which is not a Dolch word).
So the rule is calibrated on: validation in-vocab attempts (want: accept)
plus programmatic corruptions of common non-Dolch kids' words (want: reject).

Rule shape (deterministic): ACCEPT if any of
  - sound distance to top candidate == 0
  - raw edit distance to top candidate <= 1
  - model margin (top1 - top2 score) >= MARGIN
Pick the smallest MARGIN that rejects >= 80% of OOV while accepting >= 90%
of in-vocab validation attempts. Report the chosen rule on the test split.
"""
import json
import random
from pathlib import Path

import numpy as np

from features import lev, pair_features, sound_form
from gen_v1_data import FAMILIES, RNG_FAMILIES
from calibrate_threshold import split
from train_v1 import load_vocab

HERE = Path(__file__).parent

OOV_WORDS = ["phone", "friend", "people", "animal", "teacher", "happy",
             "family", "elephant", "banana", "orange", "purple", "window",
             "school", "water", "brother", "sister", "mother", "father"]
# keep only words genuinely outside the vocab
OOV_WORDS = [w for w in OOV_WORDS]


def top_stats(attempt, coef, intercept, words, lvl):
    feats = np.array([pair_features(attempt, c, lvl[c]) for c in words])
    scores = feats @ coef + intercept
    order = np.argsort(-scores, kind="stable")
    top = words[order[0]]
    if attempt in set(words):
        top = attempt
        s_top = scores[words.index(attempt)]
        rest = np.delete(scores, words.index(attempt))
        margin = float(s_top - rest.max())
    else:
        margin = float(scores[order[0]] - scores[order[1]])
    return top, margin


def main():
    words, lvl = load_vocab()
    word_set = set(words)
    oov_words = [w for w in OOV_WORDS if w not in word_set]
    data = np.load(HERE / "model_v1.npz")
    coef, intercept = data["coef"], float(data["intercept"][0])
    rows = [json.loads(l) for l in (HERE / "data" / "examples_v1.jsonl").read_text().splitlines()]
    _, val, test = split(rows, words)

    rng = random.Random(7)
    oov_attempts = []
    for w in oov_words:
        for fam_name, fn in FAMILIES.items():
            outs = fn(w, rng) if fam_name in RNG_FAMILIES else fn(w)
            for v in outs:
                if v != w:
                    oov_attempts.append({"attempt": v, "target": w})
    print(f"OOV words: {oov_words} -> {len(oov_attempts)} OOV attempts")

    def stats_for(split_rows):
        out = []
        for r in split_rows:
            top, margin = top_stats(r["attempt"], coef, intercept, words, lvl)
            out.append({**r, "top": top, "margin": margin,
                        "sdist": lev(sound_form(r["attempt"]), sound_form(top)),
                        "ldist": lev(r["attempt"], top)})
        return out

    val_s = stats_for(val)
    oov_s = stats_for(oov_attempts)

    def accept(s, margin_t):
        return s["sdist"] == 0 or s["ldist"] <= 1 or s["margin"] >= margin_t

    chosen = None
    for m in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]:
        acc_in = sum(accept(s, m) for s in val_s) / len(val_s)
        rej_oov = 1 - sum(accept(s, m) for s in oov_s) / len(oov_s)
        print(f"  margin>={m}: accept in-vocab {acc_in:.3f}, reject OOV {rej_oov:.3f}")
        if acc_in >= 0.90 and rej_oov >= 0.80 and chosen is None:
            chosen = m
    if chosen is None:
        # relax: best OOV rejection subject to in-vocab accept >= 0.85
        best = None
        for m in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]:
            acc_in = sum(accept(s, m) for s in val_s) / len(val_s)
            rej_oov = 1 - sum(accept(s, m) for s in oov_s) / len(oov_s)
            if acc_in >= 0.85 and (best is None or rej_oov > best[1]):
                best = (m, rej_oov)
        chosen = best[0] if best else 3.0
        print(f"  relaxed choice: margin {chosen}")
    print(f"CHOSEN margin: {chosen}")

    test_s = stats_for(test)
    acc_test = [s for s in test_s if accept(s, chosen)]
    report = {
        "rule": "accept if sound_dist(top)==0 or edit_dist(top)<=1 or margin>=MARGIN",
        "margin": chosen,
        "calibrated_on": "validation in-vocab attempts + programmatic OOV attempts (seed 7)",
        "oov_words_used": oov_words,
        "oov_attempts": len(oov_attempts),
        "val_accept_rate_invocab": sum(accept(s, chosen) for s in val_s) / len(val_s),
        "oov_reject_rate": 1 - sum(accept(s, chosen) for s in oov_s) / len(oov_s),
        "test_accept_rate": len(acc_test) / len(test_s),
        "test_accepted_top1_precision": (sum(s["top"] == s["target"] for s in acc_test) / len(acc_test)) if acc_test else None,
        "note": "Raw logistic probability was rejected as the confidence signal: on validation its accepted precision plateaued at 0.92 (t=0.95, coverage 0.64). See calibrate_threshold.py output in threshold_v1.json history.",
    }
    print(json.dumps(report, indent=2))
    (HERE / "threshold_v1.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
