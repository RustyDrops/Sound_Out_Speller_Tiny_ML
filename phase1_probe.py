"""Sound-Out Speller — Phase 1 combined view probe (PLAN-VIEWS-SPECIALISTS.md).

VALIDATION SPLIT ONLY (v2 split, seed 20261119, replicated exactly as in
phase0_diag.py). The generated exam (exam_v2.jsonl) and CHES are NEVER
opened. This is a probe, NOT a v3 candidate: the pre-registered bar is
combined views >= control + 2.0 pts validation top-1, or the view lever is
reported weak and the plan stops.

Configurations (all trained with the v2 recipe: LogisticRegression C=4.0,
class_weight balanced, Phase 0's exact negative-sampling + identity-half
procedure, extended features computed ONCE and selected by column):
  control            base 16 features (= v2; sanity: must reproduce 0.8333)
  +V4 / +TH / +V3 / +V5   control + one new group
  combined           control + all four new groups
  view-only solos    each new group alone (plan's solo-probe protocol)
  combined + S2      combined product ranker with the S2 deterministic
                     letter-name lookup toggled on (ablation; validation
                     has n=0 letter_name rows, so any effect is structural)
Plus: per-family validation top-1 (th_fv, cluster_reduce, all families),
false-correction self-rank for the combined configuration (product must
stay 1.000; raw reported too), and an S2 component smoke check on the 7
canonical letter-name attempts (component check, NOT an evaluation set).
"""
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from features import FEATURE_NAMES, lev, sound_form
from features_phase1 import (FEATURE_NAMES_EXT, TH_NAMES, V3_NAMES, V4_NAMES,
                             V5_NAMES, pair_features_ext, s2_lookup_rank)
from phase0_diag import replicate_v2_split
from train_v1 import load_vocab

HERE = Path(__file__).parent
SEED = 20261119
OUT = {"bar_pts": 2.0}
IDX = {n: i for i, n in enumerate(FEATURE_NAMES_EXT)}
BASE_COLS = [IDX[n] for n in FEATURE_NAMES]
GROUPS = {"V4_onset_rime": V4_NAMES, "TH_th_aware": TH_NAMES,
          "V3_letter_name": V3_NAMES, "V5_family_posterior": V5_NAMES}


def main():
    t0 = time.time()
    words, lvl = load_vocab()
    word_set = set(words)
    train, val, test = replicate_v2_split()
    OUT["split_sizes"] = {"train": len(train), "val": len(val), "test": len(test)}
    print(f"split: train {len(train)} / val {len(val)} / test {len(test)}")

    # ---- training pairs: Phase 0's exact procedure, extended features ----
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

    X, y = [], []
    for r in train:
        a, t = r["attempt"], r["target"]
        X.append(pair_features_ext(a, t, lvl[t])); y.append(1)
        for c in negatives_for(a, t):
            X.append(pair_features_ext(a, c, lvl[c])); y.append(0)
    id_words = words[:]
    random.Random(SEED + 1).shuffle(id_words)
    for w in sorted(set(id_words[: len(id_words) // 2])):
        X.append(pair_features_ext(w, w, lvl[w])); y.append(1)
        for c in negatives_for(w, w):
            X.append(pair_features_ext(w, c, lvl[c])); y.append(0)
    X, y = np.array(X), np.array(y)
    print(f"training pairs: {len(y)} ({int(y.sum())} positive), "
          f"{X.shape[1]} extended features, built in {time.time()-t0:.1f}s")

    # ---- validation feature tensor (computed once) ----
    val_attempts = sorted({r["attempt"] for r in val})
    F = {a: np.array([pair_features_ext(a, c, lvl[c]) for c in words])
         for a in val_attempts}
    print(f"val tensor: {len(F)} unique attempts x {len(words)} candidates "
          f"in {time.time()-t0:.1f}s")

    def make_product_ranker(coef, intercept, cols, use_s2=False):
        def rank_fn(attempt):
            scores = F[attempt][:, cols] @ coef + intercept
            order = np.argsort(-scores, kind="stable")
            ranked = [words[i] for i in order]
            if attempt in word_set:
                ranked.remove(attempt)
                ranked = [attempt] + ranked
            elif use_s2:
                ranked = s2_lookup_rank(attempt, ranked, word_set)
            return ranked
        return rank_fn

    def make_raw_ranker(coef, intercept, cols):
        def rank_fn(attempt):
            scores = F[attempt][:, cols] @ coef + intercept
            order = np.argsort(-scores, kind="stable")
            return [words[i] for i in order]
        return rank_fn

    def evaluate(rank_fn, rows):
        t1 = t3 = 0
        per_family = defaultdict(lambda: [0, 0])
        for r in rows:
            ranked = rank_fn(r["attempt"])
            pos = ranked.index(r["target"]) if r["target"] in ranked else 999
            t1 += pos == 0
            t3 += pos < 3
            per_family[r["family"]][0] += pos == 0
            per_family[r["family"]][1] += 1
        n = len(rows)
        return {"top1": round(t1 / n, 4), "top3": round(t3 / n, 4),
                "per_family_top1": {k: round(v[0] / v[1], 4)
                                    for k, v in sorted(per_family.items())},
                "per_family_n": {k: v[1] for k, v in sorted(per_family.items())}}

    def fit_and_eval(name, cols, use_s2=False):
        clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=1000)
        clf.fit(X[:, cols], y)
        rank_fn = make_product_ranker(clf.coef_[0], float(clf.intercept_[0]),
                                      cols, use_s2=use_s2)
        res = evaluate(rank_fn, val)
        res["n_features"] = len(cols)
        res["features"] = [FEATURE_NAMES_EXT[i] for i in cols]
        print(f"{name}: val top1 {res['top1']:.4f} top3 {res['top3']:.4f} "
              f"| th_fv {res['per_family_top1'].get('th_fv')} "
              f"cluster {res['per_family_top1'].get('cluster_reduce')}")
        return res, clf

    configs = {}
    # control (sanity: must reproduce Phase 0's 0.8333 / 0.9649)
    configs["control"], _ = fit_and_eval("control", BASE_COLS)
    ctrl = configs["control"]
    OUT["control_reproduces_v2"] = (ctrl["top1"] == 0.8333 and ctrl["top3"] == 0.9649)
    print(f"control reproduces v2 validation exactly: {OUT['control_reproduces_v2']}")

    # each addition on top of control
    for gname, gcols_names in GROUPS.items():
        cols = BASE_COLS + [IDX[n] for n in gcols_names]
        configs[f"plus_{gname}"], _ = fit_and_eval(f"+{gname}", cols)
    # view-only solos for the new groups (plan's solo-probe protocol)
    for gname, gcols_names in GROUPS.items():
        cols = [IDX[n] for n in gcols_names]
        configs[f"solo_{gname}"], _ = fit_and_eval(f"solo {gname}", cols)
    # combined
    all_cols = [IDX[n] for n in FEATURE_NAMES_EXT]
    configs["combined"], comb_clf = fit_and_eval("combined", all_cols)
    # combined + S2 ablation (same fitted model, lookup toggled on)
    comb_coef, comb_int = comb_clf.coef_[0], float(comb_clf.intercept_[0])
    s2_rank = make_product_ranker(comb_coef, comb_int, all_cols, use_s2=True)
    configs["combined_plus_S2"] = evaluate(s2_rank, val)
    configs["combined_plus_S2"]["n_features"] = len(all_cols)
    print(f"combined+S2: val top1 {configs['combined_plus_S2']['top1']:.4f} "
          f"top3 {configs['combined_plus_S2']['top3']:.4f}")
    OUT["configs"] = configs

    # S2 trigger audit on validation attempts
    from features_phase1 import parse_letter_name
    trig = [a for a in val_attempts
            if a not in word_set and parse_letter_name(a, word_set) is not None]
    OUT["s2_val_triggers"] = {"n_val_attempts": len(val_attempts),
                              "triggered": trig,
                              "letter_name_val_rows": sum(1 for r in val
                                                          if r["family"] == "letter_name")}
    print(f"S2 triggers on validation: {len(trig)} of {len(val_attempts)} unique attempts")

    # S2 component smoke check (canonical attempts; component check only,
    # NOT validation and NOT the exam)
    smoke = {}
    comb_raw = make_raw_ranker(comb_coef, comb_int, all_cols)
    # NOTE: the generator's letter_name dict lists oh->o, but 'oh' is NOT in
    # the Dolch-314 vocabulary, so that pair can never be generated and the
    # vocab-checked parser correctly returns None for 'o' (recorded in the
    # smoke entry below). The real canonical pair set is 6, not 7.
    for att, target in [("r", "are"), ("y", "why"), ("u", "you"), ("c", "see"),
                        ("i", "eye"), ("b", "be"), ("o", "oh")]:
        feats = np.array([pair_features_ext(att, c, lvl[c]) for c in words])
        scores = feats @ comb_coef + comb_int
        order = np.argsort(-scores, kind="stable")
        ranked = [words[i] for i in order]
        if target not in word_set:
            product = s2_lookup_rank(att, ranked, word_set)
            note = ("target is NOT in the vocabulary (generator dict entry is "
                    "dead: the pair can never be generated); vocab-checked "
                    "parser returns None, S2 correctly does not fire")
        elif att in word_set:
            ranked.remove(att)
            product = [att] + ranked
            note = "attempt is itself a vocab word: exact-match rule claims it; S2 never fires (by design)"
        else:
            product = s2_lookup_rank(att, ranked, word_set)
            note = "S2 applied on top of combined raw ranking"
        smoke[att] = {"target": target, "combined_raw_rank_of_target":
                      ranked.index(target) + 1 if target in ranked else None,
                      "with_S2_top1": product[0], "note": note}
    OUT["s2_smoke_check_canonical"] = smoke
    OUT["s2_canonical_coverage_corrected"] = {
        "pairs_that_exist_and_S2_fires": ["r->are", "y->why", "u->you", "c->see", "b->be"],
        "pair_claimed_by_exact_match_rule": ["i->eye (attempt 'i' is itself a vocab word)"],
        "pair_that_cannot_exist": ["o->oh ('oh' is not in the Dolch-314 vocabulary; "
                                   "the generator's letter_name dict entry never fires)"],
        "note": "Phase 0 counted 7 canonical pairs from the generator's static dict and "
                "6/7 usable. The vocab-checked count is: 6 real pairs, 5 reachable by S2, "
                "1 ('i') excluded by design. S2's ceiling is therefore even smaller than "
                "Phase 0 stated — it remains a ~0-byte deterministic component, not a lever.",
    }
    for att, s in smoke.items():
        print(f"smoke '{att}' -> target {s['target']}: raw rank "
              f"{s['combined_raw_rank_of_target']}, with-S2 top1 {s['with_S2_top1']}")

    # ---- false-correction for the combined configuration ----
    def self_rank_product(use_s2):
        ok = 0
        for w in words:
            feats = np.array([pair_features_ext(w, c, lvl[c]) for c in words])
            scores = feats @ comb_coef + comb_int
            order = np.argsort(-scores, kind="stable")
            ranked = [words[i] for i in order]
            if w in word_set:
                ranked.remove(w)
                ranked = [w] + ranked
            elif use_s2:
                ranked = s2_lookup_rank(w, ranked, word_set)
            ok += ranked[0] == w
        return ok / len(words)

    def self_rank_raw():
        ok = 0
        for w in words:
            feats = np.array([pair_features_ext(w, c, lvl[c]) for c in words])
            scores = feats @ comb_coef + comb_int
            order = np.argsort(-scores, kind="stable")
            ok += words[order[0]] == w
        return ok / len(words)

    fc_prod = self_rank_product(False)
    fc_prod_s2 = self_rank_product(True)
    fc_raw = self_rank_raw()
    OUT["false_correction_combined"] = {
        "product_self_top1": fc_prod, "product_plus_S2_self_top1": fc_prod_s2,
        "raw_model_self_top1": fc_raw, "n_words": len(words),
        "invariant_holds": fc_prod == 1.0 and fc_prod_s2 == 1.0}
    print(f"false-correction combined: product {fc_prod:.4f} "
          f"(+S2 {fc_prod_s2:.4f}), raw {fc_raw:.4f}")

    # ---- the bar ----
    delta = round(100 * (configs["combined"]["top1"] - ctrl["top1"]), 2)
    OUT["combined_delta_pts_vs_control"] = delta
    OUT["verdict"] = ("PASS — combined views beat control by "
                      f"{delta:+.2f} pts (>= +2.0 bar)"
                      if delta >= OUT["bar_pts"] else
                      "WEAK — combined views gain "
                      f"{delta:+.2f} pts (< +2.0 bar); view lever reported "
                      "weak, plan stops before any v3")
    print(f"combined delta vs control: {delta:+.2f} pts -> {OUT['verdict']}")
    OUT["elapsed_s"] = round(time.time() - t0, 1)
    (HERE / "phase1_results.json").write_text(json.dumps(OUT, indent=2))
    print(f"done in {OUT['elapsed_s']}s -> phase1_results.json")


if __name__ == "__main__":
    main()
