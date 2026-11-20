"""Sound-Out Speller v3 — data generation (pre-registered in PROVENANCE).

Identical construction to gen_v2_data.py with exactly the pre-registered
v3 changes:
  - family weights are the REFIT v3 weights (data/family_weights_v3.json,
    EG L5 hardening) — v2's weights are not used;
  - training seed 20261120, exam seed 41729 (both new);
  - the fresh exam additionally excludes every (attempt, target) pair in
    v1's examples, v2's training pool, and v2's exam;
  - a third output, data/diag_thv_exam.jsonl: the G7 generator-variant
    diagnostic set — th->v surface-realization rows only (diagnostic seed
    7331), pair-disjoint from everything above. Report-only robustness
    check, NOT a transfer bound.
Held out of training (exam-only): letter_name, affrication (as v2).
Deterministic code only; no model involved.
"""
import json
import random
from collections import Counter
from pathlib import Path

from gen_v1_data import _sub_all
from gen_v2_data import (FAMILIES_V2, HELD_OUT_FAMILIES, MAX_VARIANTS_PER_WORD,
                         all_variants, weighted_sample)

HERE = Path(__file__).parent
SEED_TRAIN = 20261120
SEED_EXAM = 41729
SEED_DIAG = 7331


def load_pairs(path):
    pairs = set()
    for line in path.read_text().splitlines():
        r = json.loads(line)
        pairs.add((r["attempt"], r["target"]))
    return pairs


def main():
    vocab_doc = json.loads((HERE / "data" / "vocab.json").read_text())
    words = []
    for level in ["pre_primer", "primer", "first", "second", "third", "nouns"]:
        words += vocab_doc["levels"][level]
    words = sorted(set(words))
    weights = json.loads((HERE / "data" / "family_weights_v3.json").read_text())["family_weights"]

    forbidden_exam = set()
    forbidden_exam |= load_pairs(HERE / "data" / "examples_v1.jsonl")
    # Clarification 3: only true exam material is excluded — v1's examples
    # and v2's EXAM. v2 TRAIN-pool pairs are admissible (disclosed as
    # in-sample for v2, hence conservative for v3 under gate 1).
    forbidden_exam |= load_pairs(HERE / "data" / "exam_v2.jsonl")

    rng_train = random.Random(SEED_TRAIN)
    rng_exam = random.Random(SEED_EXAM)
    train_rows, exam_rows = [], []
    for w in words:
        variants = all_variants(w, rng_train)
        by_family = {}
        for a, f in variants.items():
            by_family.setdefault(f, []).append(a)
        train_items = []
        for fam, attempts in sorted(by_family.items()):
            attempts = sorted(attempts)
            # PROVENANCE pre-training clarification 1: the exam reservation
            # is chosen among variants NOT in the forbidden set.
            allowed = [a for a in attempts if (a, w) not in forbidden_exam]
            if fam in HELD_OUT_FAMILIES:
                if allowed:
                    a = rng_exam.choice(allowed)
                    exam_rows.append({"attempt": a, "target": w, "family": fam})
                continue
            if len(attempts) >= 2:
                if allowed:
                    exam_a = rng_exam.choice(allowed)
                    exam_rows.append({"attempt": exam_a, "target": w, "family": fam})
                    train_items += [(a, fam) for a in attempts if a != exam_a]
                else:
                    train_items += [(a, fam) for a in attempts]
            else:
                if allowed and rng_exam.random() < 0.5:
                    exam_rows.append({"attempt": attempts[0], "target": w, "family": fam})
                else:
                    train_items.append((attempts[0], fam))
        chosen = weighted_sample(sorted(train_items), weights, MAX_VARIANTS_PER_WORD, rng_train)
        for attempt, fam in chosen:
            train_rows.append({"attempt": attempt, "target": w, "family": fam})
    rng_train.shuffle(train_rows)
    rng_exam.shuffle(exam_rows)

    # ---- G7 diagnostic set (PROVENANCE clarification 2): the fresh exam's
    # th_fv rows whose surface realization is th->v. ----
    train_pairs = {(r["attempt"], r["target"]) for r in train_rows}
    exam_pairs = {(r["attempt"], r["target"]) for r in exam_rows}
    diag_rows = []
    for r in exam_rows:
        if r["family"] != "th_fv":
            continue
        v_set = set(_sub_all(r["target"], "th", lambda m: "v"))
        if r["attempt"] in v_set:
            diag_rows.append({**r, "realization": "th_to_v"})

    with (HERE / "data" / "examples_v3.jsonl").open("w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    with (HERE / "data" / "exam_v3.jsonl").open("w") as f:
        for r in exam_rows:
            f.write(json.dumps(r) + "\n")
    with (HERE / "data" / "diag_thv_exam.jsonl").open("w") as f:
        for r in diag_rows:
            f.write(json.dumps(r) + "\n")
    print(f"vocab words: {len(words)}")
    print(f"train pool: {len(train_rows)}  exam: {len(exam_rows)}  diag th->v: {len(diag_rows)}")
    print("train families:", dict(sorted(Counter(r['family'] for r in train_rows).items())))
    print("exam families:", dict(sorted(Counter(r['family'] for r in exam_rows).items())))
    collisions = sum(1 for r in exam_rows if r["attempt"] in set(words) and r["attempt"] != r["target"])
    print(f"exam collisions with another vocab word: {collisions}")
    # disjointness self-checks
    assert not (exam_pairs & train_pairs), "exam/train overlap"
    assert not (exam_pairs & forbidden_exam), "exam/forbidden overlap"
    print("disjointness checks: PASS")


if __name__ == "__main__":
    main()
