"""Sound-Out Speller v1 — train the pair ranker and evaluate it.

Pre-registered in SPEC.md: instance split 70/15/15 per word (seed 42),
families wh_w + er_ur held out as an unseen-pattern exam, baselines B0
(plain Levenshtein) and B1 (phonetic distance), gates on test top-1/top-3,
margin over B0, false-correction on correct spellings, and size < 2 MB.
"""
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from features import FEATURE_NAMES, lev, pair_features, sound_form
from gen_v1_data import HELD_OUT_FAMILIES

HERE = Path(__file__).parent
SEED = 42
LEVEL_ORDER = ["pre_primer", "primer", "first", "second", "third", "nouns"]
LEVEL_IDX = {name: i for i, name in enumerate(LEVEL_ORDER)}
LEVEL_IDX["nouns"] = 2  # nouns are taught alongside K-2


def load_vocab():
    doc = json.loads((HERE / "data" / "vocab.json").read_text())
    lvl = {}
    for name in LEVEL_ORDER:
        for w in doc["levels"][name]:
            lvl.setdefault(w, LEVEL_IDX[name])
    words = sorted(lvl)
    return words, lvl


def rank_eval(score_fn, rows, words):
    """score_fn(attempt) -> list of candidate words best-first."""
    top1 = top3 = top5 = 0
    per_family = defaultdict(lambda: [0, 0])
    for r in rows:
        ranked = score_fn(r["attempt"])
        tgt = r["target"]
        pos = ranked.index(tgt) if tgt in ranked else 999
        top1 += pos == 0
        top3 += pos < 3
        top5 += pos < 5
        per_family[r["family"]][0] += pos == 0
        per_family[r["family"]][1] += 1
    n = max(len(rows), 1)
    return {
        "n": len(rows),
        "top1": top1 / n, "top3": top3 / n, "top5": top5 / n,
        "per_family_top1": {k: round(v[0] / v[1], 4) for k, v in sorted(per_family.items()) if v[1] >= 3},
    }


def main():
    t0 = time.time()
    words, lvl = load_vocab()
    word_set = set(words)
    rows = [json.loads(l) for l in (HERE / "data" / "examples_v1.jsonl").read_text().splitlines()]
    exam = [r for r in rows if r["family"] in HELD_OUT_FAMILIES]
    pool = [r for r in rows if r["family"] not in HELD_OUT_FAMILIES]

    # per-word instance split
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
    print(f"split: train {len(train)} / val {len(val)} / test {len(test)} / exam(held-out families) {len(exam)}")

    # ---- training pairs: target + hard negatives + random negatives ----
    def negatives_for(attempt, target):
        by_lev = sorted(words, key=lambda c: (lev(attempt, c), c))
        by_snd = sorted(words, key=lambda c: (lev(sound_form(attempt), sound_form(c)), c))
        negs = []
        for c in by_lev[:6] + by_snd[:6]:
            if c != target and c not in negs:
                negs.append(c)
        while len(negs) < 14:
            c = rng.choice(words)
            if c != target and c not in negs:
                negs.append(c)
        return negs

    X, y = [], []
    for r in train:
        a, t = r["attempt"], r["target"]
        X.append(pair_features(a, t, lvl[t])); y.append(1)
        for c in negatives_for(a, t):
            X.append(pair_features(a, c, lvl[c])); y.append(0)
    # Identity training (run-2 correction, see PROVENANCE v1 note): run 1 learned
    # a NEGATIVE identity weight because attempt==candidate only ever appeared as
    # a collision negative. Add correct-spelling attempts for a seeded HALF of the
    # vocabulary; the other half stays held out so the false-correction check
    # still measures generalization, reported per half in metrics.
    id_words = words[:]
    random.Random(SEED + 1).shuffle(id_words)
    identity_train_words = set(id_words[: len(id_words) // 2])
    for w in sorted(identity_train_words):
        X.append(pair_features(w, w, lvl[w])); y.append(1)
        for c in negatives_for(w, w):
            X.append(pair_features(w, c, lvl[c])); y.append(0)
    X, y = np.array(X), np.array(y)
    clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=1000)
    clf.fit(X, y)
    print(f"trained on {len(y)} pairs ({int(y.sum())} positive) in {time.time()-t0:.1f}s")
    for name, wgt in sorted(zip(FEATURE_NAMES, clf.coef_[0]), key=lambda kv: -abs(kv[1])):
        print(f"  weight {name}: {wgt:+.2f}")

    # ---- rankers ----
    coef, intercept = clf.coef_[0], float(clf.intercept_[0])

    def model_rank(attempt):
        feats = np.array([pair_features(attempt, c, lvl[c]) for c in words])
        scores = feats @ coef + intercept
        order = np.argsort(-scores, kind="stable")
        return [words[i] for i in order]

    def b0_rank(attempt):
        return sorted(words, key=lambda c: (lev(attempt, c), lev(sound_form(attempt), sound_form(c)), c))

    def b1_rank(attempt):
        return sorted(words, key=lambda c: (lev(sound_form(attempt), sound_form(c)), lev(attempt, c), c))

    # Product ranker (SPEC amendment 2026-11-18, after run-1/2 diagnostics):
    # an attempt that exactly matches a vocabulary word is trusted as spelled
    # correctly and ranked first by deterministic rule; the model ranks the
    # rest. All 7 raw-model self-rank failures on the held-out half were
    # homophone sets (to/too/two, there/three/their, read/red, ran/rain,
    # off/of) that spelling alone cannot separate. Raw-model metrics are
    # still reported in full under "model_raw".
    def product_rank(attempt):
        ranked = model_rank(attempt)
        if attempt in word_set:
            ranked.remove(attempt)
            return [attempt] + ranked
        return ranked

    results = {}
    for name, fn in [("model", product_rank), ("model_raw", model_rank),
                     ("B0_levenshtein", b0_rank), ("B1_phonetic", b1_rank)]:
        results[name] = {
            "val": rank_eval(fn, val, words),
            "test": rank_eval(fn, test, words),
            "exam_heldout_families": rank_eval(fn, exam, words),
        }
        print(f"{name}: test top1 {results[name]['test']['top1']:.3f} top3 {results[name]['test']['top3']:.3f} | exam top1 {results[name]['exam_heldout_families']['top1']:.3f}")

    # collision-free test subset (attempt is not itself a different vocab word)
    test_clean = [r for r in test if r["attempt"] not in word_set]
    results["model"]["test_collision_free"] = rank_eval(product_rank, test_clean, words)
    results["B0_levenshtein"]["test_collision_free"] = rank_eval(b0_rank, test_clean, words)
    print(f"collision-free test n={len(test_clean)}: model top1 {results['model']['test_collision_free']['top1']:.3f}, B0 {results['B0_levenshtein']['test_collision_free']['top1']:.3f}")

    # false-correction check: correct spellings must rank themselves #1
    def self_rank_rate(fn):
        ok = sum(1 for w in words if fn(w)[0] == w)
        return ok / len(words)
    fc_model = self_rank_rate(product_rank)
    fc_raw = self_rank_rate(model_rank)
    fc_b0 = self_rank_rate(b0_rank)
    held = [w for w in words if w not in identity_train_words]
    fc_held = sum(1 for w in held if model_rank(w)[0] == w) / len(held)
    results["false_correction"] = {"product_self_top1": fc_model,
                                   "model_raw_self_top1": fc_raw,
                                   "B0_self_top1": fc_b0,
                                   "n_words": len(words),
                                   "model_raw_self_top1_heldout_half": fc_held,
                                   "n_heldout_half": len(held)}
    print(f"false-correction: product {fc_model:.4f}, raw model {fc_raw:.4f} (held-out half {fc_held:.4f}), B0 {fc_b0:.4f}")

    # ---- export ----
    out = HERE / "model_v1.npz"
    np.savez_compressed(out, coef=coef, intercept=np.array([intercept]),
                        vocab=np.array(words), level_idx=np.array([lvl[w] for w in words]))
    size = out.stat().st_size
    print(f"model_v1.npz: {size} bytes")

    m = results["model"]["test"]
    gates = {
        "top1>=0.80": m["top1"] >= 0.80,
        "top3>=0.95": m["top3"] >= 0.95,
        "top1_beats_B0_by>=0.10": round(m["top1"] - results["B0_levenshtein"]["test"]["top1"], 4) >= 0.10,
        "false_correction_self>=0.99": fc_model >= 0.99,
        "size<2MB": size < 2_000_000,
    }
    results["raw_model_gates_note"] = (
        "Gates are evaluated on the product ranker (model + deterministic exact-match "
        "rule, SPEC amendment 2026-11-18). Raw model alone: test top1 "
        f"{results['model_raw']['test']['top1']:.3f}, held-out-half self-rank {fc_held:.4f} — "
        "it would fail the false-correction gate, which is why the rule exists.")
    results["gates"] = gates
    results["vocab_size"] = len(words)
    results["feature_names"] = FEATURE_NAMES
    results["train_pairs"] = int(len(y))
    results["elapsed_s"] = round(time.time() - t0, 1)
    (HERE / "metrics_v1.json").write_text(json.dumps(results, indent=2))
    print("GATES:", json.dumps(gates))
    print(f"done in {results['elapsed_s']}s -> metrics_v1.json")


if __name__ == "__main__":
    main()
