"""Sound-Out Speller C1b — train + evaluation (pre-registered, frozen).

v3 pattern ONLY (train_v3.py): LogisticRegression C=4.0, balanced,
hard-confuser + sampled negatives, identity-half, combined Phase-1 views
(47 features) + S2 letter-name lookup as the product. Family weights are
the v3 weights behind gen_c1b_data.py, reused unchanged. The ONLY lever
vs v3 is the target vocabulary (frozen 1,000 words, data/vocab_c1b.json).

Scores, in one sitting on the identical fresh exam (seed 90210,
pair-disjoint per gen_c1b_data.py): C1b product, B0 (plain Levenshtein),
and v3 (its own Dolch-314 vocab; expanded-vocab targets it cannot rank
score 0 — disclosed in metrics). Plus: Dolch-314 slice, per-family
scores, false-correction (v3 product definition, S2 on/off, raw model
also reported), and the Birkbeck-filtered-to-expanded-targets exam
(eval-only, OTA terms, never shipped, NOT CHES).

Gates are frozen in autonomous/c1b-preregistration.md — copied verbatim
into the gate code below. CHES gate: RETIRED (Rusty, 2026-11-19),
cited as retired, never passed.
"""
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from features import lev, pair_features, sound_form
from features_phase1 import FEATURE_NAMES_EXT, pair_features_ext, s2_lookup_rank
from train_v1 import load_vocab

HERE = Path(__file__).parent
SEED = 20261201


def split_c1b(rows):
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


def eval_rows(rank_of, rows):
    """rank_of(attempt) -> ranked candidate list (cached by caller)."""
    top1 = top3 = top5 = 0
    per_family = defaultdict(lambda: [0, 0])
    for r in rows:
        ranked = rank_of(r["attempt"])
        tgt = r["target"]
        pos = ranked.index(tgt) if tgt in ranked else 999
        top1 += pos == 0
        top3 += pos < 3
        top5 += pos < 5
        per_family[r["family"]][0] += pos == 0
        per_family[r["family"]][1] += 1
    n = max(len(rows), 1)
    return {"n": len(rows), "top1": top1 / n, "top3": top3 / n, "top5": top5 / n,
            "per_family_top1": {k: round(v[0] / v[1], 4) for k, v in sorted(per_family.items())},
            "per_family_n": {k: v[1] for k, v in sorted(per_family.items())}}


def main():
    t0 = time.time()
    words, lvl = load_vocab()
    word_set = set(words)
    assert len(words) == 1000
    dolch_doc = json.loads((HERE / "data" / "vocab.json").read_text())
    dolch_words = sorted({w for v in dolch_doc["levels"].values() for w in v})
    dolch_set = set(dolch_words)
    assert len(dolch_words) == 314 and dolch_set <= word_set

    rows = [json.loads(l) for l in (HERE / "data" / "examples_c1b.jsonl").read_text().splitlines()]
    exam = [json.loads(l) for l in (HERE / "data" / "exam_c1b.jsonl").read_text().splitlines()]
    gen_meta = json.loads((HERE / "data" / "c1b_gen_meta.json").read_text())
    train, val, test = split_c1b(rows)
    print(f"split: train {len(train)} / val {len(val)} / test {len(test)} / fresh exam {len(exam)}", flush=True)

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

    Xb_all, yb_all = [], []
    for r in train:
        a, t = r["attempt"], r["target"]
        Xb_all.append(pair_features_ext(a, t, lvl[t])); yb_all.append(1)
        for c in negatives_for(a, t):
            Xb_all.append(pair_features_ext(a, c, lvl[c])); yb_all.append(0)
    id_words = words[:]
    random.Random(SEED + 1).shuffle(id_words)
    for w in sorted(set(id_words[: len(id_words) // 2])):
        Xb_all.append(pair_features_ext(w, w, lvl[w])); yb_all.append(1)
        for c in negatives_for(w, w):
            Xb_all.append(pair_features_ext(w, c, lvl[c])); yb_all.append(0)
    X = np.array(Xb_all); y = np.array(yb_all)
    print(f"trained pool: {len(y)} pairs ({int(y.sum())} positive) in {time.time()-t0:.1f}s", flush=True)

    clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=1000)
    clf.fit(X, y)
    coef, inter = clf.coef_[0], float(clf.intercept_[0])

    # ---- rankers (per-attempt caches; features computed once per attempt) ----
    def make_ranker(cands, cand_lvl, coef_, inter_, use_s2):
        cand_set = set(cands)
        cache = {}

        def rank_of(attempt):
            if attempt not in cache:
                feats = np.array([pair_features_ext(attempt, c, cand_lvl[c]) for c in cands])
                scores = feats @ coef_ + inter_
                order = np.argsort(-scores, kind="stable")
                ranked = [cands[i] for i in order]
                if attempt in cand_set:
                    ranked.remove(attempt)
                    ranked = [attempt] + ranked
                elif use_s2:
                    ranked = s2_lookup_rank(attempt, ranked, cand_set)
                cache[attempt] = ranked
            return cache[attempt]

        return rank_of

    c1b_rank = make_ranker(words, lvl, coef, inter, use_s2=True)

    def b0_rank_of(attempt):
        return sorted(words, key=lambda c: (lev(attempt, c), lev(sound_form(attempt), sound_form(c)), c))

    # v3 on the expanded exam: v3's own model + Dolch-314 vocab/features.
    d3 = np.load(HERE / "model_v3.npz", allow_pickle=False)
    v3_words = [str(w) for w in d3["vocab"]]
    assert v3_words == dolch_words, "v3 model vocab must be the Dolch-314 in sorted order"
    v3_lvl = {w: int(d3["level_idx"][i]) for i, w in enumerate(v3_words)}
    v3_rank = make_ranker(v3_words, v3_lvl, d3["coef"], float(d3["intercept"][0]), use_s2=True)

    results = {"split": {"train": len(train), "val": len(val), "test": len(test)},
               "train_pairs": int(len(y)), "exam_n": len(exam), "gen_meta": gen_meta}

    fe_c1b = eval_rows(c1b_rank, exam)
    fe_b0 = eval_rows(b0_rank_of, exam)
    fe_v3 = eval_rows(v3_rank, exam)
    dolch_slice = [r for r in exam if r["target"] in dolch_set]
    fe_c1b_dolch = eval_rows(c1b_rank, dolch_slice)
    results["fresh_exam"] = {"c1b_product": fe_c1b, "B0_levenshtein": fe_b0,
                             "v3_product_own_vocab": fe_v3,
                             "c1b_dolch314_slice": fe_c1b_dolch,
                             "v3_note": "v3 ranks over its Dolch-314 vocab only; targets outside it score 0 on the expanded exam."}
    results["val_c1b"] = eval_rows(c1b_rank, val)
    results["dev_test_c1b"] = eval_rows(c1b_rank, test)
    print(f"FRESH EXAM n={len(exam)}: C1b {fe_c1b['top1']:.4f} | B0 {fe_b0['top1']:.4f} | v3 {fe_v3['top1']:.4f} | Dolch slice {fe_c1b_dolch['top1']:.4f}", flush=True)
    print(f"  per-family C1b: {fe_c1b['per_family_top1']}", flush=True)
    print(f"  per-family B0: {fe_b0['per_family_top1']}", flush=True)

    # ---- false-correction (v3 product definition; raw model reported too) ----
    def self_rank_product():
        ok = 0
        for w in words:
            ranked = c1b_rank(w)  # exact-match rule inside the product ranker
            ok += ranked[0] == w
        return ok / len(words)

    def self_rank_raw():
        ok = 0
        for w in words:
            feats = np.array([pair_features_ext(w, c, lvl[c]) for c in words])
            scores = feats @ coef + inter
            ok += words[int(np.argmax(scores))] == w
        return ok / len(words)

    fc_product = self_rank_product()
    fc_raw = self_rank_raw()
    results["false_correction"] = {"product_self_top1": fc_product,
                                   "product_note": "v3 product definition: exact-match claims the attempt; S2 never fires on a vocab-word attempt, so S2 on == S2 off by construction (as in v3).",
                                   "raw_model_self_top1": fc_raw, "n_words": len(words)}
    print(f"false-correction: product {fc_product:.4f}, raw model {fc_raw:.4f}", flush=True)

    # ---- Birkbeck-filtered exam (eval-only, NOT CHES, never shipped) ----
    def norm(s):
        return s.lower().replace("'", "").replace("-", "")

    pairs = []
    cur = None
    for line in (HERE / "data" / "birkbeck_eval_only" / "missp.dat").read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("$"):
            cur = norm(line[1:])
        elif cur is not None:
            a = norm(line)
            if cur in word_set and a and a.isalpha() and a != cur:
                pairs.append({"attempt": a, "target": cur, "family": "birkbeck"})
    bb_c1b = eval_rows(c1b_rank, pairs)
    bb_b0 = eval_rows(b0_rank_of, pairs)
    results["birkbeck_filtered"] = {"n_pairs": len(pairs),
                                    "n_unique_attempts": len({p["attempt"] for p in pairs}),
                                    "c1b_top1": bb_c1b["top1"], "B0_top1": bb_b0["top1"],
                                    "note": "Eval-only (OTA item 0643 terms), never shipped, NOT CHES."}
    print(f"BIRKBECK filtered n={len(pairs)}: C1b {bb_c1b['top1']:.4f} | B0 {bb_b0['top1']:.4f}", flush=True)

    # ---- export C1b model (same format as v3) ----
    out_path = HERE / "model_c1b.npz"
    np.savez_compressed(out_path, coef=coef, intercept=np.array([inter]),
                        vocab=np.array(words), level_idx=np.array([lvl[w] for w in words]),
                        feature_names=np.array(FEATURE_NAMES_EXT))
    size = out_path.stat().st_size
    demo_vocab_bytes = len(json.dumps(words).encode())
    results["model_size_bytes"] = size
    results["demo_model_json_vocab_array_bytes"] = demo_vocab_bytes
    print(f"model_c1b.npz: {size} bytes; demo vocab array: {demo_vocab_bytes} bytes", flush=True)

    # ---- frozen gates (c1b-preregistration.md, verbatim numbers) ----
    fam = fe_c1b["per_family_top1"]
    gates = {
        "1_fresh_exam_top1>=B0+0.10_AND_dolch_slice>=0.881": bool(
            fe_c1b["top1"] >= fe_b0["top1"] + 0.10 and fe_c1b_dolch["top1"] >= 0.881),
        "2_false_correction_0 (product self-rank == 1.000, S2 on and off)": bool(fc_product >= 1.0),
        "3_family_floors (th_fv >= 0.286 AND cluster_reduce >= 0.577)": bool(
            fam.get("th_fv", 0) >= 0.286 and fam.get("cluster_reduce", 0) >= 0.577),
        "4_birkbeck_filtered C1b >= B0": bool(bb_c1b["top1"] >= bb_b0["top1"]),
        "5_size_<=_16KB": bool(size <= 16 * 1024),
        "6_CHES": "RETIRED (Rusty, 2026-11-19) — cited as retired, never passed, not evaluated.",
    }
    results["gates"] = gates
    results["beats_v3_on_expanded_exam"] = bool(fe_c1b["top1"] > fe_v3["top1"])
    all_pass = all(v for v in gates.values() if isinstance(v, bool))
    results["all_gates_pass"] = bool(all_pass)
    results["verdict"] = (
        "PASS — all frozen gates pass" + (" and C1b beats v3 on the expanded exam." if results["beats_v3_on_expanded_exam"] else " BUT C1b does not beat v3 on the expanded exam, so no promotion.")
        if all_pass else "FAIL — at least one frozen gate failed; C1b rejected, v3 stays default.")
    results["disclosure"] = ("Generated-exam only — never scored on real children's misspellings. "
                             "CHES gate RETIRED (2026-11-19), never passed. Birkbeck exam is eval-only, not CHES.")
    results["elapsed_s"] = round(time.time() - t0, 1)
    (HERE / "metrics_c1b.json").write_text(json.dumps(results, indent=2))
    print("GATES:", json.dumps(gates))
    print("VERDICT:", results["verdict"])
    print(f"done in {results['elapsed_s']}s -> metrics_c1b.json")
    print("TRAIN_C1B_DONE")


if __name__ == "__main__":
    main()
