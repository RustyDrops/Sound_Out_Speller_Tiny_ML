"""Calibrate the demo's low-confidence threshold on the VALIDATION split.

For each val attempt, take the product ranker's top candidate and its logistic
probability. Choose the smallest threshold whose accepted-set precision
(top-1 correct among accepted) is >= 0.95, maximizing coverage. The chosen
threshold is then reported (not tuned) on the test split.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from features import pair_features
from gen_v1_data import HELD_OUT_FAMILIES
from train_v1 import load_vocab

HERE = Path(__file__).parent
SEED = 42


def split(rows, words):
    pool = [r for r in rows if r["family"] not in HELD_OUT_FAMILIES]
    by_word = defaultdict(list)
    for r in pool:
        by_word[r["target"]].append(r)
    rng = random.Random(SEED)
    train, val, test = [], [], []
    for w, rs in by_word.items():
        rs = rs[:]
        rng.shuffle(rs)
        n = len(rs)
        n_tr = max(1, int(round(n * 0.70))) if n > 2 else n - 1 if n == 2 else 1
        n_va = int(round(n * 0.15)) if n > 2 else 0
        train += rs[:n_tr]
        val += rs[n_tr:n_tr + n_va]
        test += rs[n_tr + n_va:]
    return train, val, test


def top_info(attempt, coef, intercept, words, lvl):
    feats = np.array([pair_features(attempt, c, lvl[c]) for c in words])
    scores = feats @ coef + intercept
    order = np.argsort(-scores, kind="stable")
    top = words[order[0]]
    if attempt in set(words):
        top = attempt
    p = 1.0 / (1.0 + np.exp(-scores[order[0]]))
    return top, float(p)


def main():
    words, lvl = load_vocab()
    data = np.load(HERE / "model_v1.npz")
    coef, intercept = data["coef"], float(data["intercept"][0])
    rows = [json.loads(l) for l in (HERE / "data" / "examples_v1.jsonl").read_text().splitlines()]
    _, val, test = split(rows, words)

    def collect(split_rows):
        out = []
        for r in split_rows:
            top, p = top_info(r["attempt"], coef, intercept, words, lvl)
            out.append({"attempt": r["attempt"], "target": r["target"], "top": top, "p": p,
                        "correct": top == r["target"]})
        return out

    val_info = collect(val)
    print(f"val n={len(val_info)}")
    chosen = None
    for t in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        acc = [v for v in val_info if v["p"] >= t]
        if not acc:
            continue
        prec = sum(v["correct"] for v in acc) / len(acc)
        cov = len(acc) / len(val_info)
        print(f"  t={t:.2f}: accepted {len(acc)} (coverage {cov:.3f}), precision {prec:.3f}")
        if prec >= 0.95 and chosen is None:
            chosen = t
    if chosen is None:
        chosen = 0.95
        print("  no threshold reached 0.95 precision; falling back to 0.95")
    print(f"CHOSEN threshold: {chosen}")

    test_info = collect(test)
    acc = [v for v in test_info if v["p"] >= chosen]
    rej = [v for v in test_info if v["p"] < chosen]
    report = {
        "threshold": chosen,
        "chosen_on": "validation, smallest t with accepted precision >= 0.95",
        "val_n": len(val_info),
        "test_n": len(test_info),
        "test_accepted": len(acc),
        "test_accepted_precision": (sum(v["correct"] for v in acc) / len(acc)) if acc else None,
        "test_rejected": len(rej),
        "test_rejected_would_have_been_right": sum(v["correct"] for v in rej),
    }
    print(json.dumps(report, indent=2))
    (HERE / "threshold_v1.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
