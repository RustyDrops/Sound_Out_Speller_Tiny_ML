"""Sound-Out Speller v3 — REFIT of generator family weights (EG L5 hardening).

Mandatory before any v3 training (EG-REVIEW-2026-11-19, finding L5,
accepted-with-changes; disposition in PROVENANCE). v2's fitted weights are
NOT reused. Same deterministic edit-operation classifier, same frozen
category->family mapping, same source file as fit_weights_v2.py — the ONLY
changes are the hardened exclusion rules:

  Firewall (v2, kept): a pair is fitted on only if its TARGET is not in the
  Dolch-314 vocabulary (the real exam is CHES filtered to Dolch targets, so
  every exam row has a Dolch target).

  NEW — attempt-overlap exclusion (EG L5 (i), stricter reading): a fitting
  pair is ALSO excluded if its normalized attempt string appears among the
  attempt strings of ANY Dolch-target row in the amalgamated file. The
  exclusion set is built from every parsed Dolch-target row BEFORE any
  other filter is applied (no alpha filter, no t != a filter), because an
  attempt string that appears as a recorded misspelling of a Dolch word
  could constitute exam material regardless of how its own row would have
  been filtered. Normalization = lowercase, matching the v2 fit's handling.

CHES rows are not separately identifiable in the amalgamation and are never
used as an exam here. Local fitting only; nothing derived ships.
Output: data/family_weights_v3.json.
"""
import json
import re
from collections import Counter
from pathlib import Path

from fit_weights_v2 import MAP, ALL_FAMILIES, classify, load_vocab_words, parse_birkbeck

HERE = Path(__file__).parent


def main():
    vocab = load_vocab_words()
    pairs = parse_birkbeck(HERE / "data" / "birkbeck_eval_only" / "missp.dat")

    # Attempt-overlap exclusion set: normalized attempts of ALL Dolch-target
    # rows, built before any other filtering (stricter reading, see header).
    dolch_attempts = set()
    for target, attempt in pairs:
        if target.lower() in vocab:
            dolch_attempts.add(attempt.lower())

    kept, skipped_dolch, skipped_overlap, skipped_form = [], 0, 0, 0
    for target, attempt in pairs:
        t, a = target.lower(), attempt.lower()
        if not re.fullmatch(r"[a-z]+", t) or not re.fullmatch(r"[a-z]+", a):
            skipped_form += 1
            continue
        if t == a:
            skipped_form += 1
            continue
        if t in vocab:
            skipped_dolch += 1
            continue  # v2 FIREWALL: no Dolch target is ever fitted on
        if a in dolch_attempts:
            skipped_overlap += 1
            continue  # NEW (EG L5): attempt string appears on a Dolch-target row
        kept.append((t, a))
    cats = Counter(classify(t, a) for t, a in kept)
    total = sum(cats.values())
    weights = {f: 0.0 for f in ALL_FAMILIES}
    for cat, n in cats.items():
        share = n / total
        for fam in MAP[cat]:
            weights[fam] += share / len(MAP[cat])
    s = sum(weights.values())
    weights = {f: round(w / s, 6) for f, w in sorted(weights.items())}
    v2w = json.loads((HERE / "data" / "family_weights_v2.json").read_text())
    out = {
        "source": "Birkbeck amalgamation (Mitton), author's download; OTA 0643; research use only, no redistribution",
        "fit": "v3 REFIT under EG L5 hardening (attempt-overlap exclusion added to the v2 Dolch-target firewall); v2 weights NOT reused",
        "pairs_total_in_file": len(pairs),
        "dolch_target_attempt_strings_in_exclusion_set": len(dolch_attempts),
        "pairs_used_for_fitting": len(kept),
        "pairs_skipped_dolch_target_firewall": skipped_dolch,
        "pairs_skipped_attempt_overlap_exclusion_NEW": skipped_overlap,
        "pairs_skipped_form_filters": skipped_form,
        "v2_pairs_used_for_fitting": v2w["pairs_used_for_fitting"],
        "category_counts": dict(sorted(cats.items())),
        "category_proportions": {k: round(v / total, 6) for k, v in sorted(cats.items())},
        "family_weights": weights,
        "family_weights_v2_for_comparison": v2w["family_weights"],
    }
    (HERE / "data" / "family_weights_v3.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
