"""Sound-Out Speller C1b — data generation (pre-registered, frozen).

Exactly the gen_v3_data.py construction with ONLY the C1b changes from
autonomous/c1b-preregistration.md:
  - target vocab = data/vocab_c1b.json (frozen 1,000 words) via the
    bundle's train_v1.load_vocab (Dolch levels verbatim + additions at
    level prior index 2);
  - family weights = data/family_weights_v3.json REUSED UNCHANGED;
  - seeds SEED_TRAIN_C1B = 20261201, SEED_EXAM_C1B = 90210;
  - exam (attempt, target) pairs are pair-disjoint from the union of ALL
    prior pools (examples_v1, exam_v2, examples_v2, exam_v3,
    examples_v3) and from the C1b train pool; excluded exam candidates
    are counted, not silently dropped.
Held out of training (exam-only): letter_name, affrication (as v2/v3).
Deterministic code only; no model involved.
"""
import json
import random
from collections import Counter
from pathlib import Path

from gen_v2_data import HELD_OUT_FAMILIES, MAX_VARIANTS_PER_WORD, all_variants, weighted_sample
from train_v1 import load_vocab

HERE = Path(__file__).parent
SEED_TRAIN = 20261201
SEED_EXAM = 90210

PRIOR_POOLS = ["examples_v1.jsonl", "exam_v2.jsonl", "examples_v2.jsonl",
               "exam_v3.jsonl", "examples_v3.jsonl"]


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
            allowed = [a for a in attempts if (a, w) not in forbidden_exam]
            dropped_exam_candidates += len(attempts) - len(allowed)
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

    with (HERE / "data" / "examples_c1b.jsonl").open("w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    with (HERE / "data" / "exam_c1b.jsonl").open("w") as f:
        for r in exam_rows:
            f.write(json.dumps(r) + "\n")

    train_pairs = {(r["attempt"], r["target"]) for r in train_rows}
    exam_pairs = {(r["attempt"], r["target"]) for r in exam_rows}
    assert not (exam_pairs & train_pairs), "exam/train overlap"
    assert not (exam_pairs & forbidden_exam), "exam/prior-pool overlap"
    meta = {
        "vocab_words": len(words),
        "seed_train": SEED_TRAIN, "seed_exam": SEED_EXAM,
        "train_pool": len(train_rows), "exam": len(exam_rows),
        "prior_pool_pairs_union": len(forbidden_exam),
        "exam_candidates_dropped_for_disjointness": dropped_exam_candidates,
        "train_families": dict(sorted(Counter(r["family"] for r in train_rows).items())),
        "exam_families": dict(sorted(Counter(r["family"] for r in exam_rows).items())),
        "disjointness_checks": "PASS (exam vs C1b train pool; exam vs union of all prior pools)",
    }
    (HERE / "data" / "c1b_gen_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print("GEN_C1B_DONE")


if __name__ == "__main__":
    main()
