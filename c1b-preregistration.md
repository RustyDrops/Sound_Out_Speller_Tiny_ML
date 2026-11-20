# Speller C1b — PRE-REGISTRATION (frozen 2026-11-20, before any C1b training)

Decision: `c1b-DECISION-2026-11-20.md` (approved by Rusty, decision delegated
to Ann, 2026-11-20). Prep: `c1b-prep.md`. Vocab: `data/vocab_c1b.json`
(build: `build_vocab_c1b.py`; audit: `c1b-vocab-audit.json`). Licence
evidence: `data/ngsl_ndl_source/LICENCE-EVIDENCE-2026-11-20.md`, recorded in
PROVENANCE.md (C1b section). **No training has run under this registration.**

## The candidate — one lever, data only
Identical to v3 in recipe, features, views, S2, and training pattern
(`train_v3.py` / `gen_v3_data.py` pattern: LogisticRegression C=4.0,
balanced, hard-confuser + sampled negatives, identity-half). The ONLY
change is the target vocabulary: Dolch-314 → the frozen 1,000 words of
`data/vocab_c1b.json` (314 Dolch verbatim + 613 New Dolch 1.1 in NDL rank
order + 73 NGSL 1.2 in rank order; all additions level prior index 2).
`data/vocab.json` is not overwritten.

## Gates — frozen verbatim from the decision file (ALL must pass)
1. Fresh generated exam (new seed, pair-disjoint): top-1 ≥ B0 + 0.10, AND Dolch-314 slice top-1 ≥ 0.881 (v3's 0.931 minus 5 pts non-regression floor for a 3× vocab expansion).
2. False-correction = 0: self-rank 1.000 over the full expanded vocab, S2 on and off.
3. Per-family floors: th_fv ≥ 0.286, cluster_reduce ≥ 0.577, per-family scores beside headline.
4. Birkbeck-filtered: C1b ≥ B0 on same set.
5. Size ≤ 16 KB all-in, shipped vocab accounting stated at pre-registration.
6. CHES gate: RETIRED (Rusty, 2026-11-19) — cited as retired, never passed. Disclosure stands: generated-exam only, never scored on real children's misspellings (Birkbeck is eval-only, not CHES).

Fail any gate = rejected, v3 stays GitHub default. Pass all gates + beat v3
on the expanded exam → GitHub default via PAT, v3 kept for comparison.

## Family weights
**v3 family weights reused unchanged** (`data/family_weights_v3.json`,
fitted behind the v3-hardened Birkbeck firewall). No refit — a refit with
an expanded exclusion set would be a second lever.

## Fresh generated exam — seed plan (pair-disjoint)
- New seeds, never used by v1/v2/v3 (v3 used SEED_TRAIN 20261120,
  SEED_EXAM 41729, SEED_DIAG 7331): **SEED_TRAIN_C1B = 20261201,
  SEED_EXAM_C1B = 90210**.
- Generation follows the v3 generator over the full 1,000-word vocab.
- Pair-disjointness is enforced, not assumed: the exam's (attempt, target)
  pair set must be disjoint from the union of ALL prior pools —
  `examples_v1.jsonl`, `exam_v2.jsonl`, `examples_v2.jsonl`,
  `exam_v3.jsonl`, `examples_v3.jsonl` — and from the C1b train pool.
  Overlapping pairs are dropped at generation time and the drop count is
  recorded in the run record. B0 = plain Levenshtein, scored on the
  identical exam in the same sitting. The Dolch-314 slice = exam rows
  whose target is in the original Dolch-314.

## Birkbeck-filtered exam construction (eval-only)
- Source: `data/birkbeck_eval_only/missp.dat` (Birkbeck amalgamation,
  Mitton; OTA item 0643 terms; eval-only, never shipped, never published,
  **not CHES** — CHES rows are not separately identifiable in this file
  and are never relabelled as CHES).
- Parse: `$target` header lines, following lines are attempts. Normalise
  targets and attempts with the C1b rules (lowercase, apostrophes/hyphens
  removed); keep only pairs whose normalised target is in the frozen
  1,000-word vocab; drop empty/non-alphabetic attempts and identity pairs.
- Report C1b top-1 and B0 (plain Levenshtein) top-1 on the identical
  filtered set, with the filtered pair count n stated. Gate 4: C1b ≥ B0.

## Size accounting (gate 5)
- Comparator: v3 `model_v3.npz` = 3,482 B at 314 words.
- C1b shipped vocab cost, measured at build time: 1,000 words, 4,958 total
  characters, JSON array encoding 8,958 B (`data/vocab_c1b.json` itself is
  23,655 B including metadata/audit fields — the shipped artefact carries
  only the word list + level indices, not the audit fields).
- Gate measurement: the final shipped model file (npz, same format as v3:
  coefficients + vocab + level_idx) must be ≤ 16 KB all-in; the demo
  `model.json` vocab-array bytes are counted and reported beside it.
  Headroom is real but not large — if the model file exceeds 16 KB the
  candidate fails gate 5 as registered, with no re-encoding after the fact.

## Colab CPU job plan (no run yet — this registration only)
- Route: Colab CPU exclusively (Rusty's 2026-11-20 rule supersedes the
  prep note's local-CPU estimate). CPU std runtime (~0.08 units/hr).
- One bundled zip upload: C1b train/gen scripts (v3 pattern), features,
  `data/vocab_c1b.json`, `data/family_weights_v3.json`, prior-pool pair
  files for the disjointness check, Birkbeck file for the eval-only exam.
- Balance recorded before/after; downloads (model, metrics, exam files)
  verified locally (size + sha256 + load check) BEFORE the runtime is
  deleted; runtime deleted in the same flow; `colab usage` confirms
  0 active assignments afterwards.
- Estimate: **~0.16 units** (v3-scale training ~11 min local-equivalent
  plus generation, both exams, and setup/teardown overhead).
- Attribution shipped with any resulting model: "Vocabulary includes words from the New General Service List / New Dolch List by Browne & Culligan, licensed CC BY-SA 4.0." Share-alike applies to the vocabulary list as shipped; no other project data is re-licensed.

**Status: PRE-REGISTERED, NOT RUN. 0 Colab units spent. No GitHub push.**
