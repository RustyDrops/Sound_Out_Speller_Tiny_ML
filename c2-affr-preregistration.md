# Speller C2-AFFR — PRE-REGISTRATION (frozen 2026-11-20, BEFORE any C2 training or exam scoring)

Cycle 2 candidate C2-AFFR. Keeper: **C1b** (GitHub default `3d75ca76`,
fresh generated exam 0.9434 over the frozen 1,000-word vocab).
Motivation, from `autonomous/c1b-VERDICT.md`: affrication was C1b's
weakest family — **0.1364 (n=22)** — while B0 (plain Levenshtein) scored
0.6364 on the same rows. Affrication had no gate in C1b and was held out
of training by design (as in v2/v3). This candidate tests exactly one
hypothesis: the weakness is a training-exposure gap, not a feature gap
(the V5 family-posterior feature for affrication already exists).

## The one lever — training-generator family mix ONLY

- Affrication (`tr→chr`, `dr→jr`, per Read 1975 / Treiman 1993 / Bear et
  al., implemented in `gen_v2_data.fam_affrication`, unchanged) is
  **removed from the held-out set** and enters the TRAINING generator's
  family mix. `letter_name` remains held out (exam-only), unchanged.
- **Pre-registered rate:** affrication is treated exactly like every
  other non-held-out family under the C1b construction. The generator is
  deterministic: exactly **28 of the 1,000 vocab words are
  affrication-eligible, each yielding exactly 1 variant**. For a
  single-variant family the C1b rule is an exam-seed coin flip —
  **p = 0.5 to the fresh exam, p = 0.5 to the training pool** — so the
  pre-registered training rate is **50% of eligible affrication pairs
  (expected ≈14 of 28)**, after which the unchanged weighted sampling
  (v3 family weights, affrication weight 9.3e-05 with the standard 0.01
  floor, per-word cap 14) applies as it does to all families.
- UNCHANGED from C1b: vocabulary (`data/vocab_c1b.json`, frozen 1,000),
  recipe (LogisticRegression C=4.0, class_weight balanced,
  hard-confuser + sampled negatives, identity-half), features (47,
  `features.py` + `features_phase1.py` frozen), S2, product exact-match
  rule, family weights (`data/family_weights_v3.json` reused verbatim —
  no refit), Birkbeck handling (eval-only, OTA 0643, never shipped).
- No new datasets. No frontier-model output in data, labels, filtering,
  or selection. Bans observed: no UNIPEN, no TalkBank.

## Seeds (new, never used by v1/v2/v3/C1b)

- SEED_TRAIN_C2 = **20261301** (C1b used 20261201; v3 20261120; v2 20261119)
- SEED_EXAM_C2 = **103101** (C1b used 90210; v3 41729; v2 99173)

## Fresh generated exam — construction (frozen)

Same protocol as `gen_c1b_data.py` with only the changes this lever
requires:

1. **Pair-disjointness (primary):** every exam (attempt, target) pair is
   disjoint from the C2 training pool — asserted in code. Affrication
   exam pairs are **never in C2 training** (never trained-to-the-exam):
   per eligible word, the coin flip assigns its single variant to exam
   XOR training before any sampling.
2. **Prior-pool exclusion (non-affrication families):** exam pairs are
   disjoint from the union of `examples_v1`, `exam_v2`, `examples_v2`,
   `exam_v3`, `examples_v3`, and `exam_c1b` (all prior pools C1b itself
   excluded, plus C1b's spent exam). Pairs from C1b's *training* pool
   are admissible exam material (the v3 clarification-3 pattern:
   disclosed, and conservative for C2 in any same-exam comparison
   against C1b, which trained on them).
3. **Affrication exemption — disclosed feasibility fact:** the 28
   affrication pairs are a finite deterministic space and **all 28 were
   consumed by prior exams** (22 in C1b's exam, 6 in v2's exam) while
   being held out of **all prior training**. An affrication exam that
   also excluded prior exams would have n=0 by construction and the
   candidate's central gate would be unevaluable. Affrication exam
   pairs are therefore exempt from rule 2's prior-*exam* exclusion —
   they may coincide with prior exam pairs, but they have never
   appeared in any training pool, and rule 1 (disjoint from C2
   training) is absolute. This exemption is registered here, before
   training, not discovered after scoring.
4. A generation-only dry run with the frozen seeds (no model trained,
   nothing scored) verified coverage before this registration was
   finalised: train pool 4,525 rows (affrication 10 after weighted
   sampling), exam 1,754 rows (affrication n=18, th_fv n=26,
   cluster_reduce n=86, Dolch-314 slice n=72). These counts are
   generation facts, not results.

## Gates — frozen (ALL must pass; fail any = rejected, C1b stays default)

1. Fresh generated exam top-1 **≥ 0.9434** (C1b's headline, same
   protocol, new seed, pair-disjoint as above) **AND ≥ B0 + 0.10**
   (B0 = plain Levenshtein, same exam, same sitting).
2. Dolch-314 slice top-1 **≥ 0.881** (the C1b preregistration's
   non-regression floor: v3's 0.931 − 5 pts).
3. **Affrication family top-1 ≥ 0.50** on the fresh exam, n reported
   beside it (expected n≈14–18; small-n caveat stated in the verdict).
4. False-correction = 0: product self-rank **1.000** over the full
   1,000-word vocab, **S2 on and S2 off** (both computed explicitly).
5. Family floors: **th_fv ≥ 0.286**, **cluster_reduce ≥ 0.577**
   (v2's exam numbers, carried through v3/C1b registrations).
6. Birkbeck-filtered (eval-only, NOT CHES): C2 top-1 **≥ B0** on the
   identical filtered set, n reported.
7. Size: shipped model file **≤ 16 KB** all-in (npz, C1b format);
   demo model.json vocab-array bytes counted and reported beside it.
8. CHES gate: **RETIRED** (Rusty, 2026-11-19) — cited as retired, never
   passed, not evaluated. Disclosure stands: **generated-exam only —
   never scored on real children's misspellings.**

Promotion: pass ALL gates **AND** beat C1b overall (gate 1's ≥ 0.9434
line, with the same-exam C1b score reported beside it for context) →
GitHub default via the PAT route (custom.github-repo-admin), C1b kept
for comparison, read-back verified. Otherwise no push.

## Colab plan

- Colab CPU exclusively, CPU std (~0.08 units/hr), parallel-safe with
  other assignments (recorded, not a blocker). One bundled zip via GCS
  staging, state file from the start, balance before/after.
- Downloads (model, metrics, exam/train pools, gen meta, full run log)
  verified locally — byte-size match against remote `stat`, npz load
  check (vocab 1,000, 47 coefficients), metrics parse — **before** the
  runtime is deleted; runtime deleted in the same flow; verified gone.
  On failure, the full run log is downloaded and preserved before
  deletion.
- Estimate: ~0.10 units (C1b's rate×elapsed was 0.071 for a larger exam;
  this exam is smaller).
- Attribution shipped with any resulting model: "Vocabulary includes
  words from the New General Service List / New Dolch List by Browne &
  Culligan, licensed CC BY-SA 4.0."

**Status: PRE-REGISTERED, NOT RUN. 0 Colab units spent on C2. No push.**
