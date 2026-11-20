"""Sound-Out Speller — Phase 0 diagnostics for PLAN-VIEWS-SPECIALISTS.md.

VALIDATION DATA ONLY. Uses v2's validation split, replicated exactly from
train_v2.py (examples_v2.jsonl, per-word split, seed 20261119). Loaded
models: model_v1.npz / model_v2.npz as-is. The fresh generated exam
(exam_v2.jsonl) and CHES are NEVER opened by this script. Solo-view probes
retrain the v2 recipe on the v2 TRAIN split with feature subsets — a
diagnostic control, not a candidate version.

(a) Headroom: per weak family, correct word's rank when the v2 product
    ranker errs on validation; trigger-restricted addressable/breakable
    counts for the th and cluster specialists.
(b) View solo probes: logistic rankers on single feature groups.
(c) Letter-name parse check: deterministic parser over the 7 canonical
    letter_name attempts defined in gen_v2_data.py (generator code, NOT
    the exam file; rankers are not scored on these).
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from features import FEATURE_NAMES, lev, pair_features, sound_form
from train_v1 import load_vocab
from train_v2 import make_rankers

HERE = Path(__file__).parent
SEED = 20261119
OUT = {}


def replicate_v2_split():
    rows = [json.loads(l) for l in (HERE / "data" / "examples_v2.jsonl").read_text().splitlines()]
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


def rank_of(ranked, target):
    return ranked.index(target) + 1 if target in ranked else None


def rank_bucket(rk):
    if rk == 1:
        return "rank1"
    if rk == 2:
        return "rank2"
    if rk == 3:
        return "rank3"
    if rk <= 5:
        return "rank4-5"
    if rk <= 10:
        return "rank6-10"
    return "rank>10"


def main():
    words, lvl = load_vocab()
    word_set = set(words)
    train, val, test = replicate_v2_split()
    OUT["split_sizes"] = {"train": len(train), "val": len(val), "test": len(test)}
    print(f"replicated v2 split: train {len(train)} / val {len(val)} / test {len(test)}")

    d2 = np.load(HERE / "model_v2.npz", allow_pickle=False)
    _, v2_product, _ = make_rankers(d2["coef"], float(d2["intercept"][0]), words, lvl)
    d1 = np.load(HERE / "model_v1.npz", allow_pickle=False)
    _, v1_product, _ = make_rankers(d1["coef"], float(d1["intercept"][0]), words, lvl)

    # ---------- (a) headroom on validation ----------
    def headroom(rank_fn, rows):
        errs = []
        buckets = defaultdict(int)
        for r in rows:
            ranked = rank_fn(r["attempt"])
            rk = rank_of(ranked, r["target"])
            buckets[rank_bucket(rk)] += 1
            if rk != 1:
                errs.append((r, rk))
        return errs, dict(buckets)

    v2_errs, v2_buckets = headroom(v2_product, val)
    val_top1 = 1 - len(v2_errs) / len(val)
    OUT["val_v2_product_top1"] = val_top1
    OUT["val_rank_buckets_v2"] = v2_buckets
    print(f"v2 product val top-1 {val_top1:.4f} (metrics_v2.json says 0.8333) — errors {len(v2_errs)}")
    print("rank buckets (all val):", v2_buckets)

    fam_rows = defaultdict(list)
    for r in val:
        fam_rows[r["family"]].append(r)
    per_family = {}
    for fam in ["th_fv", "cluster_reduce", "th_td", "generic_drop"]:
        rows = fam_rows.get(fam, [])
        if not rows:
            per_family[fam] = {"n_val": 0}
            continue
        errs, buckets = headroom(v2_product, rows)
        in_top3 = sum(1 for _, rk in errs if rk <= 3)
        per_family[fam] = {
            "n_val": len(rows), "errors": len(errs), "top1": 1 - len(errs) / len(rows),
            "err_rank_buckets": dict(buckets),
            "errors_with_correct_in_top3": in_top3,
            "oracle_gain_pts_if_all_top3_errors_fixed": round(100 * in_top3 / len(val), 2),
        }
        print(f"family {fam}: n={len(rows)} errors={len(errs)} in-top3={in_top3} buckets={dict(buckets)}")
    OUT["headroom_per_family"] = per_family
    OUT["letter_name_val_n"] = len(fam_rows.get("letter_name", []))
    print(f"letter_name rows in v2 val split: {OUT['letter_name_val_n']} "
          "(family held out of the v2 training pool by design — see plan)")

    # ---- trigger-restricted headroom: th specialist ----
    # Trigger (computable without the target): the attempt contains f, v, or
    # th at a site where a top-3 candidate has th/f/v/t/d — operationalized:
    # attempt contains "f" or "v" or "th", AND >=2 of the top-3 candidates
    # contain "th", "f", or "v".
    def th_trigger(attempt, ranked):
        if not any(tok in attempt for tok in ("f", "v", "th")):
            return False
        hits = sum(1 for c in ranked[:3] if any(tok in c for tok in ("th", "f", "v")))
        return hits >= 2

    def trigger_stats(trigger_fn, rows, rank_fn):
        trig = corr = addr = 0
        for r in rows:
            ranked = rank_fn(r["attempt"])
            if not trigger_fn(r["attempt"], ranked):
                continue
            trig += 1
            rk = rank_of(ranked, r["target"])
            if rk == 1:
                corr += 1
            elif rk <= 3:
                addr += 1
        return {"triggered": trig, "triggered_correct_breakable": corr,
                "triggered_errors_addressable_top3": addr,
                "oracle_gain_pts_overall_val": round(100 * addr / len(val), 2),
                "breakable_to_addressable": (round(corr / addr, 2) if addr else None)}

    th_stats = trigger_stats(th_trigger, val, v2_product)
    OUT["specialist_th_trigger_val"] = th_stats
    print("th specialist trigger:", th_stats)

    # ---- trigger-restricted headroom: cluster specialist ----
    # Trigger: attempt opens with a single consonant + vowel (reduced onset)
    # AND >=1 top-3 candidate opens with a 2-consonant cluster whose first
    # or second letter equals the attempt's first letter.
    CONS = set("bcdfgjklmnpqrstvwxz")

    def cluster_trigger(attempt, ranked):
        if len(attempt) < 2 or attempt[0] not in CONS or attempt[1] in CONS:
            return False
        for c in ranked[:3]:
            if len(c) >= 2 and c[0] in CONS and c[1] in CONS and attempt[0] in (c[0], c[1]):
                return True
        return False

    cl_stats = trigger_stats(cluster_trigger, val, v2_product)
    OUT["specialist_cluster_trigger_val"] = cl_stats
    print("cluster specialist trigger:", cl_stats)

    # ---------- (b) view solo probes ----------
    groups = {
        "phonetic_view": ["sound_lev_norm", "sound_equal", "sound_bigram_dice", "sound_first_equal"],
        "grapheme_edit_view": ["lev_norm", "bigram_dice", "trigram_dice"],
        "skeleton_view": ["vowel_skel_lev", "cons_skel_lev"],
        "edge_shape_view": ["same_first", "same_last", "prefix_ratio", "suffix_ratio", "len_diff_ratio"],
        "identity_prior_view": ["identity", "level_prior"],
        "ALL_features_control": FEATURE_NAMES[:],
    }
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
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

    Xfull, y = [], []
    for r in train:
        a, t = r["attempt"], r["target"]
        Xfull.append(pair_features(a, t, lvl[t])); y.append(1)
        for c in negatives_for(a, t):
            Xfull.append(pair_features(a, c, lvl[c])); y.append(0)
    id_words = words[:]
    random.Random(SEED + 1).shuffle(id_words)
    for w in sorted(set(id_words[: len(id_words) // 2])):
        Xfull.append(pair_features(w, w, lvl[w])); y.append(1)
        for c in negatives_for(w, w):
            Xfull.append(pair_features(w, c, lvl[c])); y.append(0)
    Xfull, y = np.array(Xfull), np.array(y)

    def eval_ranker(coef_sub, intercept, cols):
        def rank_fn(attempt):
            feats = np.array([pair_features(attempt, c, lvl[c]) for c in words])[:, cols]
            scores = feats @ coef_sub + intercept
            order = np.argsort(-scores, kind="stable")
            ranked = [words[i] for i in order]
            if attempt in word_set:
                ranked.remove(attempt)
                return [attempt] + ranked
            return ranked
        t1 = sum(1 for r in val if rank_fn(r["attempt"])[0] == r["target"]) / len(val)
        t3 = sum(1 for r in val if r["target"] in rank_fn(r["attempt"])[:3]) / len(val)
        return round(t1, 4), round(t3, 4)

    solo = {}
    for gname, feats in groups.items():
        cols = [idx[f] for f in feats]
        clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=1000)
        clf.fit(Xfull[:, cols], y)
        t1, t3 = eval_ranker(clf.coef_[0], float(clf.intercept_[0]), cols)
        solo[gname] = {"features": feats, "val_top1": t1, "val_top3": t3}
        print(f"solo {gname}: val top1 {t1:.4f} top3 {t3:.4f}")
    OUT["view_solo_probes_val"] = solo

    # ---------- (c) letter-name parser ----------
    # Letter-name spellings defined in gen_v2_data.py (generator code).
    LETTER_NAME_MAP = {"r": "are", "y": "why", "u": "you", "c": "see",
                       "i": "eye", "b": "be", "o": "oh"}
    SINGLE_LETTER_NAMES = set(LETTER_NAME_MAP)  # letters whose name is a word

    def parse_letter_name(attempt):
        """Clean parse: the whole attempt is a sequence of single letters
        each of whose letter-name is a vocabulary word (here: length-1,
        the only form the generator produces). Returns target or None."""
        if len(attempt) == 1 and attempt in LETTER_NAME_MAP:
            return LETTER_NAME_MAP[attempt]
        return None

    parsed = {a: parse_letter_name(a) for a in LETTER_NAME_MAP}
    clean = sum(1 for v in parsed.values() if v is not None)
    collisions = {a: t for a, t in LETTER_NAME_MAP.items() if a in word_set}
    OUT["letter_name_parse"] = {
        "canonical_attempts": LETTER_NAME_MAP,
        "parse_cleanly": clean, "n": len(LETTER_NAME_MAP),
        "fraction": clean / len(LETTER_NAME_MAP),
        "attempts_that_are_vocab_words_themselves": collisions,
        "note": "Attempts colliding with a vocab word (e.g. 'i') are claimed by the "
                "product exact-match rule before any specialist could run — by design.",
    }
    print(f"letter-name parse: {clean}/{len(LETTER_NAME_MAP)} clean; collisions: {collisions}")

    (HERE / "phase0_results.json").write_text(json.dumps(OUT, indent=2))
    print("wrote phase0_results.json")


if __name__ == "__main__":
    main()
