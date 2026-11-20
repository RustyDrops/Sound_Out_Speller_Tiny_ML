"""Sound-Out Speller C2-AFFR — data generation (pre-registered, frozen).

Exactly the gen_c1b_data.py construction with ONLY the C2-AFFR changes
from autonomous/c2-affr/PREREGISTRATION.md:
  - affrication is NO LONGER held out: it enters the training family mix
    under the standard non-held-out single-variant rule (exam-seed coin
    flip p=0.5 to exam / training pool) — the pre-registered 50% rate.
    letter_name remains held out (exam-only), unchanged.
  - affrication pairs are exempt from the prior-exam forbidden set (the
    28-pair deterministic space was fully consumed by prior exams while
    held out of all prior training; see preregistration rule 3). Exam
    affrication pairs are still asserted disjoint from the C2 train pool.
  - forbidden exam set for all other families = C1b's prior pools PLUS
    exam_c1b (C1b's spent exam). C1b train-pool pairs remain admissible.
  - seeds SEED_TRAIN_C2 = 20261301, SEED_EXAM_C2 = 103101 (both new).
Vocab = data/vocab_c1b.json (frozen 1,000) via the bundle's adapted
train_v1.load_vocab; family weights = data/family_weights_v3.json reused
unchanged. Deterministic code only; no model involved.
"""
import json
import random
from collections import Counter
from pathlib import Path

from gen_v2_data import MAX_VARIANTS_PER_WORD, all_variants, weighted_sample
from train_v1 import load_vocab

HERE = Path(__file__).parent
SEED_TRAIN = 20261301
SEED_EXAM = 103101
HELD_OUT_FAMILIES_C2 = {"letter_name"}

PRIOR_POOLS = ["examples_v1.jsonl", "exam_v2.jsonl", "examples_v2.jsonl",
               "exam_v3.jsonl", "examples_v3.jsonl", "exam_c1b.jsonl"]


def load_pairs(path):
    pairs = set()
    for line in path.read_text().splitlines():
        r = json.loads(line)
        pairs.add((r["attempt"], r["target"]))
    return pairs


def main():
    words, _lvl = load_vocab()
    assert len(words) == 1000, f"vocab must be the frozen 1,000, got {len(words)}"
    weights = json.loads((HERE / "data" / "family_weights_v3.json").read_text())["family_weights"]

    forbidden_exam = set()
    for name in PRIOR_POOLS:
        forbidden_exam |= load_pairs(HERE / "data" / name)

    rng_train = random.Random(SEED_TRAIN)
    rng_exam = random.Random(SEED_EXAM)
    train_rows, exam_rows = [], []
    dropped_exam_candidates = 0
    for w in words:
        variants = all_variants(w, rng_train)
        by_family = {}
        for a, f in variants.items():
            by_family.setdefault(f, []).append(a)
        train_items = []
        for fam, attempts in sorted(by_family.items()):
            attempts = sorted(attempts)
            if fam == "affrication":
                # The one lever: standard non-held-out treatment, exempt
                # from the prior-exam forbidden set (preregistration
                # rule 3). Single variant per eligible word: coin flip.
                assert len(attempts) == 1, f"affrication variant count changed for {w}: {attempts}"
                if rng_exam.random() < 0.5:
                    exam_rows.append({"attempt": attempts[0], "target": w, "family": fam})
                else:
                    train_items.append((attempts[0], fam))
                continue
            allowed = [a for a in attempts if (a, w) not in forbidden_exam]
            dropped_exam_candidates += len(attempts) - len(allowed)
            if fam in HELD_OUT_FAMILIES_C2:
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

    with (HERE / "data" / "examples_c2.jsonl").open("w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    with (HERE / "data" / "exam_c2.jsonl").open("w") as f:
        for r in exam_rows:
            f.write(json.dumps(r) + "\n")

    train_pairs = {(r["attempt"], r["target"]) for r in train_rows}
    exam_pairs = {(r["attempt"], r["target"]) for r in exam_rows}
    assert not (exam_pairs & train_pairs), "exam/train overlap"
    non_aff_exam = {(r["attempt"], r["target"]) for r in exam_rows if r["family"] != "affrication"}
    assert not (non_aff_exam & forbidden_exam), "non-affrication exam/prior-pool overlap"
    meta = {
        "candidate": "C2-AFFR",
        "vocab_words": len(words),
        "seed_train": SEED_TRAIN, "seed_exam": SEED_EXAM,
        "train_pool": len(train_rows), "exam": len(exam_rows),
        "prior_pool_pairs_union": len(forbidden_exam),
        "exam_candidates_dropped_for_disjointness": dropped_exam_candidates,
        "train_families": dict(sorted(Counter(r["family"] for r in train_rows).items())),
        "exam_families": dict(sorted(Counter(r["family"] for r in exam_rows).items())),
        "disjointness_checks": "PASS (exam vs C2 train pool, all families; non-affrication exam vs union of prior pools incl. exam_c1b; affrication exempt from prior-exam exclusion per preregistration rule 3, never in any prior training pool)",
    }
    (HERE / "data" / "c2_gen_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print("GEN_C2_DONE")


if __name__ == "__main__":
    main()
