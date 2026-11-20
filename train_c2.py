"""Sound-Out Speller C2-AFFR — train + evaluation (pre-registered, frozen).

C1b pattern ONLY (train_c1b.py): LogisticRegression C=4.0, balanced,
hard-confuser + sampled negatives, identity-half, combined Phase-1 views
(47 features) + S2 letter-name lookup as the product. Family weights are
the v3 weights behind gen_c2_data.py, reused unchanged. The ONLY lever
vs C1b is the training generator's family mix: affrication is no longer
held out (autonomous/c2-affr/PREREGISTRATION.md).

Scores, in one sitting on the identical fresh exam (seed 103101,
pair-disjoint per gen_c2_data.py): C2 product, B0 (plain Levenshtein),
and C1b product (report context; C1b trained on some admissible exam
pairs — disclosed in the preregistration — so its same-exam score is
in-sample-flavoured and conservative for C2). Plus: Dolch-314 slice,
per-family scores, false-correction (product self-rank computed with
S2 on AND S2 off from the same score cache), and the
Birkbeck-filtered-to-expanded-targets exam (eval-only, OTA terms, never
shipped, NOT CHES).

Gates are frozen in autonomous/c2-affr/PREREGISTRATION.md — copied into
the gate code below. CHES gate: RETIRED (Rusty, 2026-11-19), cited as
retired, never passed.
"""
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from features import lev, sound_form
from features_phase1 import FEATURE_NAMES_EXT, pair_features_ext, s2_lookup_rank
from train_v1 import load_vocab

HERE = Path(__file__).parent
SEED = 20261301


def split_c2(rows):
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

    rows = [json.loads(l) for l in (HERE / "data" / "examples_c2.jsonl").read_text().splitlines()]
    exam = [json.loads(l) for l in (HERE / "data" / "exam_c2.jsonl").read_text().splitlines()]
    gen_meta = json.loads((HERE / "data" / "c2_gen_meta.json").read_text())
    train, val, test = split_c2(rows)
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

    # ---- rankers: scores computed once per (model, attempt), cached ----
    def make_ranker(cands, cand_lvl, coef_, inter_, use_s2):
        cand_set = set(cands)
        score_cache = {}
        rank_cache = {}

        def scores_of(attempt):
            if attempt not in score_cache:
                feats = np.array([pair_features_ext(attempt, c, cand_lvl[c]) for c in cands])
                score_cache[attempt] = feats @ coef_ + inter_
            return score_cache[attempt]

        def rank_of(attempt):
            key = (attempt, use_s2)
            if key not in rank_cache:
                scores = scores_of(attempt)
                order = np.argsort(-scores, kind="stable")
                ranked = [cands[i] for i in order]
                if attempt in cand_set:
                    ranked.remove(attempt)
                    ranked = [attempt] + ranked
                elif use_s2:
                    ranked = s2_lookup_rank(attempt, ranked, cand_set)
                rank_cache[key] = ranked
            return rank_cache[key]

        return rank_of

    c2_rank_on = make_ranker(words, lvl, coef, inter, use_s2=True)
    c2_rank_off = make_ranker(words, lvl, coef, inter, use_s2=False)

    def b0_rank_of(attempt):
        return sorted(words, key=lambda c: (lev(attempt, c), lev(sound_form(attempt), sound_form(c)), c))

    # C1b on the same exam: C1b's own frozen model, same 1,000-word vocab.
    d1b = np.load(HERE / "model_c1b.npz", allow_pickle=False)
    c1b_words = [str(w) for w in d1b["vocab"]]
    assert c1b_words == words, "C1b model vocab must equal the frozen 1,000 in sorted order"
    c1b_lvl = {w: int(d1b["level_idx"][i]) for i, w in enumerate(c1b_words)}
    assert c1b_lvl == lvl, "C1b level indices must match the frozen vocab levels"
    c1b_rank = make_ranker(c1b_words, c1b_lvl, d1b["coef"], float(d1b["intercept"][0]), use_s2=True)

    results = {"candidate": "C2-AFFR",
               "split": {"train": len(train), "val": len(val), "test": len(test)},
               "train_pairs": int(len(y)), "exam_n": len(exam), "gen_meta": gen_meta}

    fe_c2 = eval_rows(c2_rank_on, exam)
    fe_b0 = eval_rows(b0_rank_of, exam)
    fe_c1b = eval_rows(c1b_rank, exam)
    dolch_slice = [r for r in exam if r["target"] in dolch_set]
    fe_c2_dolch = eval_rows(c2_rank_on, dolch_slice)
    results["fresh_exam"] = {"c2_product": fe_c2, "B0_levenshtein": fe_b0,
                             "c1b_product_same_exam": fe_c1b,
                             "c2_dolch314_slice": fe_c2_dolch,
                             "c1b_note": "C1b trained on some admissible exam pairs (preregistration rule 2), so its same-exam score is in-sample-flavoured; the promotion line is C1b's frozen headline 0.9434 on its own exam."}
    results["val_c2"] = eval_rows(c2_rank_on, val)
    results["dev_test_c2"] = eval_rows(c2_rank_on, test)
    print(f"FRESH EXAM n={len(exam)}: C2 {fe_c2['top1']:.4f} | B0 {fe_b0['top1']:.4f} | C1b(same exam) {fe_c1b['top1']:.4f} | Dolch slice {fe_c2_dolch['top1']:.4f}", flush=True)
    print(f"  per-family C2: {fe_c2['per_family_top1']}", flush=True)
    print(f"  per-family B0: {fe_b0['per_family_top1']}", flush=True)
    print(f"  per-family C1b: {fe_c1b['per_family_top1']}", flush=True)

    # ---- false-correction: product self-rank with S2 on AND off ----
    def self_rank(rank_of):
        ok = 0
        for w in words:
            ok += rank_of(w)[0] == w
        return ok / len(words)

    fc_on = self_rank(c2_rank_on)
    fc_off = self_rank(c2_rank_off)

    def self_rank_raw():
        ok = 0
        for w in words:
            feats = np.array([pair_features_ext(w, c, lvl[c]) for c in words])
            scores = feats @ coef + inter
            ok += words[int(np.argmax(scores))] == w
        return ok / len(words)

    fc_raw = self_rank_raw()
    results["false_correction"] = {"product_self_top1_s2_on": fc_on,
                                   "product_self_top1_s2_off": fc_off,
                                   "product_note": "Exact-match rule claims a vocab-word attempt before S2 can fire (as in v3/C1b), so on == off is expected; both were computed explicitly per the C2 preregistration.",
                                   "raw_model_self_top1": fc_raw, "n_words": len(words)}
    print(f"false-correction: product S2-on {fc_on:.4f}, S2-off {fc_off:.4f}, raw model {fc_raw:.4f}", flush=True)

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
    bb_c2 = eval_rows(c2_rank_on, pairs)
    bb_b0 = eval_rows(b0_rank_of, pairs)
    results["birkbeck_filtered"] = {"n_pairs": len(pairs),
                                    "n_unique_attempts": len({p["attempt"] for p in pairs}),
                                    "c2_top1": bb_c2["top1"], "B0_top1": bb_b0["top1"],
                                    "note": "Eval-only (OTA item 0643 terms), never shipped, NOT CHES."}
    print(f"BIRKBECK filtered n={len(pairs)}: C2 {bb_c2['top1']:.4f} | B0 {bb_b0['top1']:.4f}", flush=True)

    # ---- export C2 model (same format as C1b) ----
    out_path = HERE / "model_c2.npz"
    np.savez_compressed(out_path, coef=coef, intercept=np.array([inter]),
                        vocab=np.array(words), level_idx=np.array([lvl[w] for w in words]),
                        feature_names=np.array(FEATURE_NAMES_EXT))
    size = out_path.stat().st_size
    demo_vocab_bytes = len(json.dumps(words).encode())
    results["model_size_bytes"] = size
    results["demo_model_json_vocab_array_bytes"] = demo_vocab_bytes
    print(f"model_c2.npz: {size} bytes; demo vocab array: {demo_vocab_bytes} bytes", flush=True)

    # ---- frozen gates (c2-affr/PREREGISTRATION.md) ----
    fam = fe_c2["per_family_top1"]
    fam_n = fe_c2["per_family_n"]
    gates = {
        "1_fresh_exam_top1>=0.9434_AND_>=B0+0.10": bool(
            fe_c2["top1"] >= 0.9434 and fe_c2["top1"] >= fe_b0["top1"] + 0.10),
        "2_dolch314_slice>=0.881": bool(fe_c2_dolch["top1"] >= 0.881),
        "3_affrication>=0.50": bool(fam.get("affrication", 0) >= 0.50),
        "4_false_correction_0 (product self-rank == 1.000, S2 on and off)": bool(
            fc_on >= 1.0 and fc_off >= 1.0),
        "5_family_floors (th_fv >= 0.286 AND cluster_reduce >= 0.577)": bool(
            fam.get("th_fv", 0) >= 0.286 and fam.get("cluster_reduce", 0) >= 0.577),
        "6_birkbeck_filtered C2 >= B0": bool(bb_c2["top1"] >= bb_b0["top1"]),
        "7_size_<=_16KB": bool(size <= 16 * 1024),
        "8_CHES": "RETIRED (Rusty, 2026-11-19) — cited as retired, never passed, not evaluated.",
    }
    results["gates"] = gates
    results["affrication_exam_n"] = fam_n.get("affrication", 0)
    results["beats_c1b_headline_0.9434"] = bool(fe_c2["top1"] >= 0.9434)
    all_pass = all(v for v in gates.values() if isinstance(v, bool))
    results["all_gates_pass"] = bool(all_pass)
    results["verdict"] = (
        "PASS — all frozen gates pass and C2 beats C1b's headline (promotion condition met)."
        if all_pass and results["beats_c1b_headline_0.9434"] else
        "FAIL — at least one frozen gate failed or the C1b headline was not beaten; C2 rejected, C1b stays default.")
    results["disclosure"] = ("Generated-exam only — never scored on real children's misspellings. "
                             "CHES gate RETIRED (2026-11-19), never passed. Birkbeck exam is eval-only, not CHES.")
    results["elapsed_s"] = round(time.time() - t0, 1)
    (HERE / "metrics_c2.json").write_text(json.dumps(results, indent=2))
    print("GATES:", json.dumps(gates))
    print("VERDICT:", results["verdict"])
    print(f"done in {results['elapsed_s']}s -> metrics_c2.json")
    print("TRAIN_C2_DONE")


if __name__ == "__main__":
    main()
