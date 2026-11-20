"""Sound-Out Speller v3 — train + fresh-exam evaluation (pre-registered).

Lever: feature views ONLY — v2's frozen 16 features + the four Phase 1
groups from features_phase1.py (47 total), plus the S2 deterministic
letter-name lookup as a named, ablatable product component. Recipe
identical to v1/v2 (LogisticRegression C=4.0, balanced, same negative
sampling + identity-half procedure), training seed 20261120, family
weights = the REFIT v3 weights (data came from gen_v3_data.py).
Scores the fresh generated exam (seed 41729) for v1, v2, v3 matched;
mandatory per-component ablations on val + exam; homophone-collision and
OOV probes (report-only); false-correction; and the G7 generator-variant
diagnostic (th->v realization held out of the diagnostic model's training
rows — a robustness check, NOT a transfer bound). The CHES real-exam gate
is BLOCKED (see PROVENANCE): gate 3 is NOT EVALUATED, so v3 cannot be
declared keeper on this run regardless of generated-exam results.
"""
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from features import FEATURE_NAMES, lev, pair_features, sound_form
from features_phase1 import (FEATURE_NAMES_EXT, TH_NAMES, V3_NAMES, V4_NAMES,
                             V5_NAMES, pair_features_ext, s2_lookup_rank)
from gen_v1_data import _sub_all
from train_v1 import load_vocab, rank_eval

HERE = Path(__file__).parent
SEED = 20261120
IDX = {n: i for i, n in enumerate(FEATURE_NAMES_EXT)}
BASE_COLS = [IDX[n] for n in FEATURE_NAMES]
GROUPS = {"V4_onset_rime": V4_NAMES, "TH_th_aware": TH_NAMES,
          "V3_letter_name": V3_NAMES, "V5_family_posterior": V5_NAMES}
ALL_COLS = [IDX[n] for n in FEATURE_NAMES_EXT]


def split_v3(rows):
    by_word = defaultdict(list)
    for r in rows:
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


def main():
    t0 = time.time()
    words, lvl = load_vocab()
    word_set = set(words)
    rows = [json.loads(l) for l in (HERE / "data" / "examples_v3.jsonl").read_text().splitlines()]
    exam = [json.loads(l) for l in (HERE / "data" / "exam_v3.jsonl").read_text().splitlines()]
    diag = [json.loads(l) for l in (HERE / "data" / "diag_thv_exam.jsonl").read_text().splitlines()]
    train, val, test = split_v3(rows)
    print(f"split: train {len(train)} / val {len(val)} / test {len(test)} / fresh exam {len(exam)} / diag {len(diag)}")

    rng = random.Random(SEED)

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

    # Training pairs, keeping the pair-block of each train row (needed for
    # the G7 diagnostic subset) — v2 procedure otherwise.
    row_blocks = []  # (row, X_block, y_block)
    for r in train:
        a, t = r["attempt"], r["target"]
        Xb = [pair_features_ext(a, t, lvl[t])]
        yb = [1]
        for c in negatives_for(a, t):
            Xb.append(pair_features_ext(a, c, lvl[c])); yb.append(0)
        row_blocks.append((r, np.array(Xb), np.array(yb)))
    id_words = words[:]
    random.Random(SEED + 1).shuffle(id_words)
    id_block_X, id_block_y = [], []
    for w in sorted(set(id_words[: len(id_words) // 2])):
        id_block_X.append(pair_features_ext(w, w, lvl[w])); id_block_y.append(1)
        for c in negatives_for(w, w):
            id_block_X.append(pair_features_ext(w, c, lvl[c])); id_block_y.append(0)
    id_block = (np.array(id_block_X), np.array(id_block_y))
    X = np.vstack([b[1] for b in row_blocks] + [id_block[0]])
    y = np.concatenate([b[2] for b in row_blocks] + [id_block[1]])
    print(f"trained pool: {len(y)} pairs ({int(y.sum())} positive) in {time.time()-t0:.1f}s")

    # ---- feature tensors for every set we score ----
    def tensor(attempts, feat_fn):
        return {a: np.array([feat_fn(a, c, lvl[c]) for c in words]) for a in attempts}

    all_attempts = sorted({r["attempt"] for r in val + test + exam + diag})
    Fext = tensor(all_attempts, pair_features_ext)
    Fbase = tensor(all_attempts, pair_features)

    def product_ranker(F, cols, coef, intercept, use_s2=False):
        def fn(attempt):
            scores = F[attempt][:, cols] @ coef + intercept if cols is not None else F[attempt] @ coef + intercept
            order = np.argsort(-scores, kind="stable")
            ranked = [words[i] for i in order]
            if attempt in word_set:
                ranked.remove(attempt)
                ranked = [attempt] + ranked
            elif use_s2:
                ranked = s2_lookup_rank(attempt, ranked, word_set)
            return ranked
        return fn

    def fit(cols, Xuse=None, yuse=None):
        clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=1000)
        clf.fit((X if Xuse is None else Xuse)[:, cols], y if yuse is None else yuse)
        return clf.coef_[0], float(clf.intercept_[0])

    results = {"split": {"train": len(train), "val": len(val), "test": len(test)},
               "train_pairs": int(len(y)), "exam_n": len(exam)}

    # ---- configurations: fit once, score val + fresh exam ----
    configs = {"control_base16": BASE_COLS}
    for g, names in GROUPS.items():
        configs[f"plus_{g}"] = BASE_COLS + [IDX[n] for n in names]
    configs["combined_views_only"] = ALL_COLS
    for g, names in GROUPS.items():
        drop = {IDX[n] for n in names}
        configs[f"combined_minus_{g}"] = [c for c in ALL_COLS if c not in drop]
    fitted = {}
    abl = {}
    for name, cols in configs.items():
        coef, inter = fit(cols)
        fitted[name] = (cols, coef, inter)
        abl[name] = {
            "val": rank_eval(product_ranker(Fext, cols, coef, inter), val, words),
            "exam": rank_eval(product_ranker(Fext, cols, coef, inter), exam, words),
        }
        print(f"{name}: val {abl[name]['val']['top1']:.4f} | exam {abl[name]['exam']['top1']:.4f}")
    # combined + S2 = the v3 product configuration
    cols, coef, inter = fitted["combined_views_only"]
    v3_rank = product_ranker(Fext, cols, coef, inter, use_s2=True)
    abl["combined_plus_S2_v3_product"] = {
        "val": rank_eval(v3_rank, val, words),
        "exam": rank_eval(v3_rank, exam, words),
    }
    print(f"combined+S2 (v3 product): val {abl['combined_plus_S2_v3_product']['val']['top1']:.4f} | "
          f"exam {abl['combined_plus_S2_v3_product']['exam']['top1']:.4f}")
    results["ablations"] = abl
    results["dev_test_v3_product"] = rank_eval(v3_rank, test, words)

    # ---- v1 / v2 / B0 on the fresh exam (matched) ----
    def load_ranker(path):
        d = np.load(path, allow_pickle=False)
        assert [str(w) for w in d["vocab"]] == words
        return product_ranker(Fbase, None, d["coef"], float(d["intercept"][0]))

    v1_rank = load_ranker(HERE / "model_v1.npz")
    v2_rank = load_ranker(HERE / "model_v2.npz")

    def b0_rank(attempt):
        return sorted(words, key=lambda c: (lev(attempt, c), lev(sound_form(attempt), sound_form(c)), c))

    fe = {
        "v3_product": rank_eval(v3_rank, exam, words),
        "v3_views_only": abl["combined_views_only"]["exam"],
        "v2_product": rank_eval(v2_rank, exam, words),
        "v1_product": rank_eval(v1_rank, exam, words),
        "B0_levenshtein": rank_eval(b0_rank, exam, words),
    }
    results["fresh_exam"] = fe
    print(f"FRESH EXAM n={len(exam)}: v3 {fe['v3_product']['top1']:.3f} | v2 {fe['v2_product']['top1']:.3f} | "
          f"v1 {fe['v1_product']['top1']:.3f} | B0 {fe['B0_levenshtein']['top1']:.3f}")
    print(f"  per-family v3: {fe['v3_product']['per_family_top1']}")
    print(f"  per-family v2: {fe['v2_product']['per_family_top1']}")

    # ---- false-correction (v3 product, S2 on and off) ----
    def self_rank(use_s2):
        ok = 0
        for w in words:
            feats = np.array([pair_features_ext(w, c, lvl[c]) for c in words])
            scores = feats @ coef + inter
            order = np.argsort(-scores, kind="stable")
            ranked = [words[i] for i in order]
            ranked.remove(w)
            ranked = [w] + ranked
            if use_s2:
                pass  # attempt is a vocab word: exact-match claims it, S2 never fires (by design)
            ok += ranked[0] == w
        return ok / len(words)

    def self_rank_raw():
        ok = 0
        for w in words:
            feats = np.array([pair_features_ext(w, c, lvl[c]) for c in words])
            scores = feats @ coef + inter
            order = np.argsort(-scores, kind="stable")
            ok += words[order[0]] == w
        return ok / len(words)

    fc_on = self_rank(True)
    fc_off = self_rank(False)
    fc_raw = self_rank_raw()
    results["false_correction"] = {"product_self_top1_S2_on": fc_on,
                                   "product_self_top1_S2_off": fc_off,
                                   "raw_model_self_top1": fc_raw, "n_words": len(words)}
    print(f"false-correction: product S2-on {fc_on:.4f} / S2-off {fc_off:.4f}, raw {fc_raw:.4f}")

    # ---- probe (i): homophone / collision rows in the fresh exam ----
    coll = [r for r in exam if r["attempt"] in word_set and r["attempt"] != r["target"]]
    if coll:
        results["probe_homophone_collisions"] = {
            "n": len(coll),
            "v3_product": rank_eval(v3_rank, coll, words),
            "v2_product": rank_eval(v2_rank, coll, words),
            "v1_product": rank_eval(v1_rank, coll, words),
            "note": "Product exact-match rule trusts the attempt as correctly spelled, so "
                    "collision rows are expected to score ~0 top-1 by design; report-only.",
        }
        print(f"probe collisions n={len(coll)}: v3 {results['probe_homophone_collisions']['v3_product']['top1']:.3f}")

    # ---- probe (ii): OOV attempts + unchanged v1 confidence rule ----
    from gen_v1_data import FAMILIES as V1F, RNG_FAMILIES as V1RNG
    oov_words = ["phone", "friend", "people", "animal", "teacher", "happy", "family",
                 "elephant", "banana", "orange", "purple", "window", "school",
                 "water", "brother", "sister", "mother", "father"]
    oov_words = [w for w in oov_words if w not in word_set]
    orng = random.Random(7)
    oov_attempts = []
    for w in oov_words:
        vs = {}
        for fam_name, fn in V1F.items():
            outs = fn(w, orng) if fam_name in V1RNG else fn(w)
            for v in outs:
                if v != w and v not in vs:
                    vs[v] = fam_name
        for v in sorted(vs)[:6]:
            oov_attempts.append(v)

    def accept_rate(score_fn):
        acc = 0
        for a in oov_attempts:
            scores = score_fn(a)
            order = np.argsort(-scores, kind="stable")
            top = words[order[0]]
            margin = float(scores[order[0]] - scores[order[1]])
            if lev(sound_form(a), sound_form(top)) == 0 or lev(a, top) <= 1 or margin >= 1.5:
                acc += 1
        return acc / max(len(oov_attempts), 1)

    d1 = np.load(HERE / "model_v1.npz", allow_pickle=False)
    d2 = np.load(HERE / "model_v2.npz", allow_pickle=False)
    results["probe_oov"] = {
        "n_attempts": len(oov_attempts),
        "v3_confidence_rule_accept_rate": accept_rate(
            lambda a: np.array([pair_features_ext(a, c, lvl[c]) for c in words]) @ coef + inter),
        "v2_confidence_rule_accept_rate": accept_rate(
            lambda a: np.array([pair_features(a, c, lvl[c]) for c in words]) @ d2["coef"] + float(d2["intercept"][0])),
        "v1_confidence_rule_accept_rate": accept_rate(
            lambda a: np.array([pair_features(a, c, lvl[c]) for c in words]) @ d1["coef"] + float(d1["intercept"][0])),
        "note": "Acceptance uses the UNCHANGED v1 demo confidence rule (threshold_v1.json). "
                "Lower acceptance on OOV attempts = better rejection. Report-only probe.",
    }
    print(f"probe OOV n={len(oov_attempts)}: accept v3 {results['probe_oov']['v3_confidence_rule_accept_rate']:.3f}, "
          f"v2 {results['probe_oov']['v2_confidence_rule_accept_rate']:.3f}, "
          f"v1 {results['probe_oov']['v1_confidence_rule_accept_rate']:.3f}")

    # ---- G7 generator-variant diagnostic (report-only) ----
    def is_thv(row):
        return row["family"] == "th_fv" and row["attempt"] in set(_sub_all(row["target"], "th", lambda m: "v"))

    diag_blocks = [(r, Xb, yb) for (r, Xb, yb) in row_blocks if not is_thv(r)]
    n_removed = len(row_blocks) - len(diag_blocks)
    Xd = np.vstack([b[1] for b in diag_blocks] + [id_block[0]])
    yd = np.concatenate([b[2] for b in diag_blocks] + [id_block[1]])
    dcoef, dinter = fit(ALL_COLS, Xd, yd)
    diag_rank = product_ranker(Fext, ALL_COLS, dcoef, dinter, use_s2=True)
    results["g7_generator_variant_diagnostic"] = {
        "label": "Robustness check against rule memorization (EG G7) — NOT a transfer bound.",
        "held_out_realization": "th->v (all th->v training rows removed from the diagnostic model)",
        "train_rows_removed": n_removed,
        "n_diag_rows": len(diag),
        "diag_model_top1": rank_eval(diag_rank, diag, words)["top1"] if diag else None,
        "v3_product_top1_same_rows": rank_eval(v3_rank, diag, words)["top1"] if diag else None,
        "v2_product_top1_same_rows": rank_eval(v2_rank, diag, words)["top1"] if diag else None,
        "note": "v2 and v3 product trained on th->v pairs in their own pools, so their "
                "scores on these rows are not realization-unseen; the claim rests on the "
                "diagnostic-model contrast (PROVENANCE clarification 2).",
    }
    print(f"G7 diagnostic: removed {n_removed} th->v train rows; diag n={len(diag)}: "
          f"diag model {results['g7_generator_variant_diagnostic']['diag_model_top1']}, "
          f"v3 {results['g7_generator_variant_diagnostic']['v3_product_top1_same_rows']}, "
          f"v2 {results['g7_generator_variant_diagnostic']['v2_product_top1_same_rows']}")

    # ---- export v3 product model (combined views; S2 is a code rule) ----
    out_path = HERE / "model_v3.npz"
    np.savez_compressed(out_path, coef=coef, intercept=np.array([inter]),
                        vocab=np.array(words), level_idx=np.array([lvl[w] for w in words]),
                        feature_names=np.array(FEATURE_NAMES_EXT))
    size = out_path.stat().st_size
    results["model_size_bytes"] = size
    print(f"model_v3.npz: {size} bytes")

    # ---- gates (frozen in PROVENANCE v3 pre-registration) ----
    fam3 = fe["v3_product"]["per_family_top1"]
    gates = {
        "1_fresh_exam_top1_v3>=v1_AND_v3>=v2": bool(
            fe["v3_product"]["top1"] >= fe["v1_product"]["top1"]
            and fe["v3_product"]["top1"] >= fe["v2_product"]["top1"]),
        "2_false_correction_stays_0 (product self-rank == 1.000, S2 on and off)": bool(
            fc_on >= 1.0 and fc_off >= 1.0),
        "3_real_exam_CHES (v3 >= v1 AND v3 >= B0+0.05)": None,  # BLOCKED — see provenance
        "4_family_floors (th_fv >= 0.286 AND cluster_reduce >= 0.577)": bool(
            fam3.get("th_fv", 0) >= 0.286 and fam3.get("cluster_reduce", 0) >= 0.577),
        "5_size_<=_16KB": bool(size <= 16 * 1024),
    }
    results["gates"] = gates
    results["real_exam_ches"] = {
        "status": "BLOCKED — standalone CHES file not obtainable legitimately on 2026-11-19 "
                  "(OTA handle resolver HTTP 500; author's own download offers only the "
                  "amalgamated Birkbeck file in which CHES rows are not separately "
                  "identifiable). Gate 3 NOT EVALUATED and therefore NOT PASSED. "
                  "No substitute real corpus scored.",
        "v1_top1": None, "v3_top1": None, "B0_top1": None,
    }
    gen_pass = all(v for k, v in gates.items() if v is not None)
    results["verdict"] = (
        ("Generated-exam gates PASS (1, 2, 4, 5); " if gen_pass else
         "Generated-exam gates FAIL (see gates table); ")
        + "KEEPER DEFERRED to CHES — gate 3 (real CHES exam) is blocked/not "
          "evaluated, so v3 cannot be declared keeper on this run regardless "
          "of generated-exam results. v1 remains the keeper.")
    results["elapsed_s"] = round(time.time() - t0, 1)
    (HERE / "metrics_v3.json").write_text(json.dumps(results, indent=2))
    print("GATES:", json.dumps(gates))
    print("VERDICT:", results["verdict"])
    print(f"done in {results['elapsed_s']}s -> metrics_v3.json")


if __name__ == "__main__":
    main()
