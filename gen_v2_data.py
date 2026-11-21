"""Sound-Out Speller v2 — generator realism (the ONE lever, pre-registered).

Same v1 families (imported unchanged) plus five new families from the
published developmental taxonomy (Read/Treiman/Bear):
  nasal_liquid_omit, cluster_reduce, letter_name, affrication, th_fv.

Outputs:
  data/examples_v2.jsonl — training pool: NON-held-out families only,
    variants selected per word (cap 14, as v1) by seeded sampling weighted
    by the off-exam fitted family weights (floor 0.01, see PROVENANCE).
  data/exam_v2.jsonl — fresh generated exam: ALL families incl. held-out,
    stratified (<=1 variant per family per word), exam seed 99173, and
    pair-disjoint from the v2 training pool and from v1's examples.

Held out of training (exam-only): letter_name, affrication — mirroring
v1's wh_w/er_ur holdout. Deterministic code only; no model involved.
"""
import json
import random
import re
from pathlib import Path

from gen_v1_data import FAMILIES as V1_FAMILIES, RNG_FAMILIES, _sub_all

HERE = Path(__file__).parent
SEED_TRAIN = 20261119
SEED_EXAM = 99173
MAX_VARIANTS_PER_WORD = 14
HELD_OUT_FAMILIES = {"letter_name", "affrication"}
WEIGHT_FLOOR = 0.01

SONORANTS = "mnlr"
CONS = set("bcdfgjklmnpqrstvwxz")


def fam_nasal_liquid_omit(w):
    out = []
    for i in range(1, len(w) - 1):
        if w[i] in SONORANTS and w[i + 1] in CONS and w[i - 1] in "aeiou":
            out.append(w[:i] + w[i + 1:])
    return out


def fam_cluster_reduce(w):
    out = []
    if len(w) >= 3 and w[0] in CONS and w[1] in CONS:
        out.append(w[1:])            # drop first element (stop -> top)
        out.append(w[0] + w[2:])     # drop second element (play -> pay)
        if len(w) >= 4 and w[2] in CONS:
            out.append(w[0] + w[2:])         # 3-cluster: drop middle (street -> street w/o r? s+t+r -> s+t)
            out.append(w[:2] + w[3:])        # 3-cluster: drop third (str -> st)
    return list(dict.fromkeys(v for v in out if v and v != w))


def fam_letter_name(w):
    return {"are": ["r"], "why": ["y"], "you": ["u"], "see": ["c"],
            "eye": ["i"], "be": ["b"], "oh": ["o"]}.get(w, [])


def fam_affrication(w):
    return _sub_all(w, "tr", lambda m: "chr") + _sub_all(w, "dr", lambda m: "jr")


def fam_th_fv(w):
    return _sub_all(w, "th", lambda m: "f") + _sub_all(w, "th", lambda m: "v")


NEW_FAMILIES = {
    "nasal_liquid_omit": fam_nasal_liquid_omit,
    "cluster_reduce": fam_cluster_reduce,
    "letter_name": fam_letter_name,
    "affrication": fam_affrication,
    "th_fv": fam_th_fv,
}
FAMILIES_V2 = {**V1_FAMILIES, **NEW_FAMILIES}


def all_variants(word, rng):
    variants = {}
    for fam_name, fn in FAMILIES_V2.items():
        outs = fn(word, rng) if fam_name in RNG_FAMILIES else fn(word)
        for v in outs:
            if v != word and v not in variants:
                variants[v] = fam_name
    return variants  # attempt -> family


def weighted_sample(items, weights, cap, rng):
    """items: list of (attempt, family). Sample without replacement with
    probability proportional to family weight (floored, renormalized)."""
    if len(items) <= cap:
        return items[:]
    pool = items[:]
    chosen = []
    while pool and len(chosen) < cap:
        ws = [max(weights.get(fam, 0.0), WEIGHT_FLOOR) for _, fam in pool]
        tot = sum(ws)
        r = rng.random() * tot
        acc = 0.0
        for idx, wgt in enumerate(ws):
            acc += wgt
            if r <= acc:
                chosen.append(pool.pop(idx))
                break
    return chosen


def main():
    vocab_doc = json.loads((HERE / "data" / "vocab.json").read_text())
    words = []
    for level in ["pre_primer", "primer", "first", "second", "third", "nouns"]:
        words += vocab_doc["levels"][level]
    words = sorted(set(words))
    weights = json.loads((HERE / "data" / "family_weights_v2.json").read_text())["family_weights"]

    v1_pairs = set()
    for line in (HERE / "data" / "examples_v1.jsonl").read_text().splitlines():
        r = json.loads(line)
        v1_pairs.add((r["attempt"], r["target"]))

    rng_train = random.Random(SEED_TRAIN)
    rng_exam = random.Random(SEED_EXAM)
    train_rows, exam_rows = [], []
    for w in words:
        variants = all_variants(w, rng_train)
        # Split per family BEFORE training sampling (PROVENANCE amendment 3):
        # >=2 variants -> 1 reserved for exam; exactly 1 -> exam-seed coin
        # flip; held-out families -> exam only. Exam pairs are also excluded
        # if they appeared in v1's examples.
        by_family = {}
        for a, f in variants.items():
            by_family.setdefault(f, []).append(a)
        train_items = []
        for fam, attempts in sorted(by_family.items()):
            attempts = sorted(attempts)
            if fam in HELD_OUT_FAMILIES:
                a = rng_exam.choice(attempts)
                if (a, w) not in v1_pairs:
                    exam_rows.append({"attempt": a, "target": w, "family": fam})
                continue
            if len(attempts) >= 2:
                exam_a = rng_exam.choice(attempts)
                if (exam_a, w) not in v1_pairs:
                    exam_rows.append({"attempt": exam_a, "target": w, "family": fam})
                train_items += [(a, fam) for a in attempts if a != exam_a]
            else:
                if rng_exam.random() < 0.5 and (attempts[0], w) not in v1_pairs:
                    exam_rows.append({"attempt": attempts[0], "target": w, "family": fam})
                else:
                    train_items.append((attempts[0], fam))
        chosen = weighted_sample(sorted(train_items), weights, MAX_VARIANTS_PER_WORD, rng_train)
        for attempt, fam in chosen:
            train_rows.append({"attempt": attempt, "target": w, "family": fam})
    rng_train.shuffle(train_rows)
    rng_exam.shuffle(exam_rows)
    with (HERE / "data" / "examples_v2.jsonl").open("w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    with (HERE / "data" / "exam_v2.jsonl").open("w") as f:
        for r in exam_rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(f"vocab words: {len(words)}")
    print(f"train pool: {len(train_rows)}  exam: {len(exam_rows)}")
    print("train families:", dict(sorted(Counter(r['family'] for r in train_rows).items())))
    print("exam families:", dict(sorted(Counter(r['family'] for r in exam_rows).items())))
    collisions = sum(1 for r in exam_rows if r["attempt"] in set(words) and r["attempt"] != r["target"])
    print(f"exam collisions with another vocab word: {collisions}")


if __name__ == "__main__":
    main()
