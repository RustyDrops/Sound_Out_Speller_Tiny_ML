# Sound-Out Speller — Provenance
Built: 2026-11-18 · Status: local v1 — NOT published or uploaded anywhere.
Nothing publishes without Rusty's approval of the exact version.

## Dataset
| Field | Entry |
|---|---|
| Name | Dolch word list — 220 service words (pre-primer, primer, first, second, third grade) + 95 nouns |
| Source URL | https://www.kidsworksheetsfree.com/sight-words/dolch-list/ (prints the full list; retrieved 2026-11-18) |
| Creator / citation | Edward William Dolch, compiled 1936, revised 1948, from high-frequency words in early reading |
| License | The source states the words are common words and the compilation "has long been treated as public domain" and prints it as a free reference. Word lists are factual compilations; no expressive text is reproduced. Logged 2026-11-18 |
| Contents | `data/vocab.json`: 220 service words by level + nouns. Normalizations: `don't` → `dont`, `good-bye` → `goodbye`, `Christmas` lowercased, multiword `Santa Claus` excluded. Final unique count is printed by `gen_v1_data.py` and recorded in `metrics_v1.json` |
| Use here | The bounded target vocabulary the ranker chooses among, and the base words from which misspellings are generated |

## Synthetic data
| Field | Entry |
|---|---|
| Generator | Programmatic script `gen_v1_data.py` (deterministic phonics-error rules, seed 42) — no model of any kind generated, labelled, scored, or filtered any example |
| Families | silent_e_drop, vowel_team_swap, ai_e, double_single, single_double, ck_k, c_k, wh_w (held out), th_td, vowel_swap, unstressed_drop, transpose, y_i, er_ur (held out), generic_drop |

## Frontier-model contact — affirmation
No Muse / Muse, OpenAI / Luna, or Google / Gemini output was used as training data,
labels, scores, rankings, filtering, or dataset selection for this model. The vocabulary
is a published 1936/1948 word list retrieved from the source above; all misspellings are
produced by the deterministic code generator. Gemini Decision 1 is reserved for
benchmark-only comparison on held-out sets under Rusty's 2026-11-18 ruling and plays
no part in training. No personal data was used.

## Method (pre-registered in SPEC.md before training)
- Splits: per-word instance split train 70% / val 15% / test 15% (seed 42);
  families wh_w and er_ur held out of training as an unseen-pattern exam;
  correct-spelling false-correction check over the full vocabulary.
- Features: ~15 pair features (raw + phonetic Levenshtein, char n-gram Dice,
  vowel/consonant skeleton distances, edge/prefix/suffix matches, length diff).
- Classifier: Logistic Regression on (attempt, candidate) match pairs; negatives are
  hard confusers (nearest by edit/phonetic distance) plus sampled vocabulary words.
- Baselines: B0 plain Levenshtein ranking; B1 phonetic-distance ranking.
- Gates: see SPEC.md (top-1 ≥ 0.80, top-3 ≥ 0.95, ≥ +0.10 over B0 top-1,
  false-correction self-rank ≥ 99%, size < 2 MB).

## Results — v1 (2026-11-18, local CPU on this VM, ~19k training pairs, fit in seconds; full run incl. evaluation ~72 s)
Full numbers: `metrics_v1.json`. Run-1 diagnostic preserved in `metrics_v1_run1_diagnostic.json`.
Disclosure (also in SPEC.md amendment): the development test informed two design fixes
(identity training + exact-match rule), so these are development numbers; v2 must be
judged on a fresh exam.

- Vocabulary: 314 unique words · generated examples: 1,549 (34 collide with another vocab word)
- Splits: train 1,122 / val 273 / test 120 / held-out-family exam 34 (wh_w 7, er_ur 27 — small n)
- **Product ranker (model + exact-match rule), test: top-1 0.867, top-3 0.975**
- Raw model alone, test: top-1 0.883, top-3 0.975
- Baselines, test: B0 Levenshtein top-1 0.783 / top-3 0.942 · B1 phonetic top-1 0.767
- Collision-free test (n=116): product top-1 0.897 vs B0 0.810
- Held-out-family exam: product top-1 0.912 (er_ur 0.926, wh_w 0.857) — baselines also 0.912 at this small n
- False-correction: product 1.0000 (by rule) · raw model 0.9682 overall, 0.9554 on the
  identity-held-out half — all 7 held-out failures were homophone sets
  (to/too/two, there/three/their, read/red, ran/rain, off/of)
- Weak test families (small n): th_td 0.556, generic_drop 0.600, vowel_team_swap 0.625, double_single 0.750
- Model file: `model_v1.npz` 2,521 bytes · runtime `predict_v1.py` (pure NumPy)
- Gates (SPEC.md): top-1 ≥0.80 ✓ · top-3 ≥0.95 ✓ · false-correction ≥0.99 ✓ (product) ·
  size <2 MB ✓ · **margin over B0 ≥ +0.10 ✗ — product +0.083 (raw model exactly +0.100).**
  The exact-match rule deliberately spends the collision guesses (4 test cases) to buy
  zero false-corrections. 4 of 5 gates pass; the margin gate fails by that named tradeoff.
- Smoke test (`predict_v1.py`): wos→was ✓, teh→the ✓, becos→because ✓, sed→seed (said #2),
  riting→ring (write missed), jumpt→jump, fone→done (phone not in vocab — no correct
  answer exists; v1 has no low-confidence "not in this level's words" threshold yet).

## Demo + release (2026-11-18, at Rusty's direction)
- Browser demo `demo/index.html`: one self-contained file (model JSON + JS port of
  features.py inlined by `build_demo.py`; loads nothing from the network). JS/Python
  parity verified with `parity_test.js`: 30/30 attempts identical (top-5, exact flag,
  margin, accepted). Rendered states screenshot-checked at 390×844 and 360×740.
- Confidence rule: the raw logistic probability failed as a confidence signal
  (validation accepted-precision plateaued at 0.92), so the demo's "not in this
  level's words yet" state uses the deterministic rule calibrated in
  `calibrate_threshold2.py` (sound distance 0, or edit distance ≤1, or margin ≥1.5):
  accepts 94.9% of in-vocab validation attempts, rejects 80.5% of programmatic
  out-of-vocab attempts. Full record in `threshold_v1.json`.
- GitHub: published PUBLIC at Rusty's instruction ("push to github", following the
  Scam Shield / Write It Right pattern) —
  https://github.com/RustyDrops/Sound_Out_Speller_Tiny_ML, v1 commit b8792cf2,
  verified by read-back (metrics and binary model SHA-checked via the connector).
  The repo's social-preview image is a manual Settings upload (promo card is in
  `promo/`); a push cannot set it.

## Known limitations
See SPEC.md "Honest limits": no -ed pattern words in Dolch, homophones need context,
generated errors approximate — but do not prove performance on — real children's spelling.

---

# v2 — PRE-REGISTRATION (written 2026-11-19 BEFORE any v2 training, fitting, or exam scoring)

Lineup item C1 (`LINEUP-2026-11-19.md`). Local CPU only, 0 Colab units.

## The one lever
Generator realism ONLY. Model architecture, feature set (`features.py`), classifier
recipe (LogisticRegression C=4.0, class_weight balanced, same negative-sampling and
identity-half procedure as v1), vocabulary (Dolch-314, unchanged — vocab growth is
C1b, a separate later version), and the product exact-match rule are all UNCHANGED
from v1. Only the misspelling generator changes:
- NEW families on the published developmental taxonomy (Read 1975; Treiman 1993;
  Bear et al., *Words Their Way* — published findings, implemented as code rules,
  no license barrier): `nasal_liquid_omit` (sonorant m/n/l/r dropped before a
  consonant: plant→plat, hand→had), `cluster_reduce` (initial-cluster reduction:
  drop the liquid/second element or the initial s: play→pay, stop→top),
  `letter_name` (whole-word letter-name spellings for vocab words that are letter
  names: are→r, why→y, you→u, see→c, eye→i, be→b, oh→o), `affrication`
  (tr→chr, dr→jr), `th_fv` (th→f, th→v, beside the existing th→t/d family).
- All v1 families retained. `letter_name` and `affrication` are HELD OUT of v2
  training entirely (exam-only families, mirroring v1's wh_w/er_ur holdout) so the
  fresh exam also measures unseen-pattern generalization.
- Family sampling weights: v1 generated every variant up to a per-word cap; v2
  selects variants per word (same cap, 14) with seeded sampling weighted by family
  weights FITTED OFF-EXAM (next section). Training seed 20261119 (new).

## Weight fitting — leakage firewall (CRITICAL, frozen)
- Family weights are estimated by a deterministic edit-operation classifier run
  over Birkbeck (Mitton) misspelling/target pairs whose **target word is NOT in
  the Dolch-314 vocabulary**, plus the published-taxonomy presence of the new
  families. Because the real exam (below) is CHES filtered to Dolch-314 targets,
  every exam row has a Dolch target; excluding all Dolch targets from fitting
  guarantees no exam row, and no exam target word, was used in fitting.
- CHES itself is never used in training, fitting, filtering, or selection, in any form.

## Birkbeck / CHES source + terms (verified at source 2026-11-19)
- Corpus: Birkbeck Spelling Error Corpus, Roger Mitton (1985), Oxford Text Archive
  item 0643 (hdl 20.500.12024/0643). OTA catalogue lists it "Publicly Available";
  OTA standard terms = research use, no redistribution → EVAL/FITTING ONLY here,
  locally; no Birkbeck/CHES data or derived data ships, is published, or is
  committed to any repo.
- Acquisition status: the OTA handle resolver returned HTTP 500 through the
  browser fetch (retries exhausted) and the OTA header file was unreachable, so
  the OTA package with its SEPARATE per-source files (incl. the standalone CHES
  file) could NOT be obtained today. What WAS obtained legitimately is Mitton's
  own download page (dcs.bbk.ac.uk/~ROGER/corpora.html, the author's own
  distribution of this corpus), whose `birkbeck` file is an AMALGAMATION of the
  native-speaker Birkbeck files (36,133 misspellings of 6,136 words) — CHES rows
  are inside it but are NOT separately identifiable. Consequence, recorded before
  any scoring: amalgamated rows will be used ONLY for off-exam weight fitting
  under the Dolch-target firewall above. They will NOT be relabelled as CHES.

## Exams (frozen)
- (a) Fresh generated exam: generated by the v2 generator with a SEPARATE exam
  seed (99173), excluding every (attempt, target) pair present in v2
  train/val/test and in v1's examples. Scored for BOTH the v1 product ranker
  (model_v1.npz, loaded as-is) and the v2 product ranker — matched same-exam
  comparison. Per-family top-1 reported for both.
- (b) Real exam: CHES filtered to Dolch-314 targets, scored for v1, v2, and B0
  (plain Levenshtein ranking, v1 definition). **Status at pre-registration:
  BLOCKED** — the standalone CHES file could not be obtained legitimately today
  (see acquisition status above). If it remains unobtainable, gate 3 below is
  recorded as NOT EVALUATED and v2 cannot be declared keeper on this run,
  whatever the generated-exam numbers say. No substitute real corpus will be
  used without authorization.
- Adversarial probes (named, report-only unless a gate says otherwise):
  (i) homophone/collision probe — exam rows whose attempt is itself a different
  vocabulary word (to/too/two-type collisions); top-1 reported for v1 and v2;
  (ii) out-of-vocab probe — rule-generated attempts at non-vocabulary words
  (phone, elephant, school, …); the v1 demo confidence rule (threshold_v1.json:
  sound distance 0, or edit distance ≤ 1, or margin ≥ 1.5) acceptance rate is
  reported for v1 and v2 (the ranker itself has no rejection; this measures the
  shipped confidence rule, unchanged).

## Gates (ALL must pass for v2 to be keeper; frozen)
1. Fresh generated exam (a): v2 product top-1 ≥ v1 product top-1 on the SAME exam.
2. False-correction stays 0: product ranker ranks every correctly spelled
   vocabulary word #1 for itself (self-rank rate = 1.000, exact-match rule).
3. Real exam (b): v2 CHES top-1 ≥ v1 CHES top-1 AND v2 CHES top-1 ≥ B0 CHES
   top-1 + 0.05. (Blocked = not passed; see exam (b) status.)
- Confidence rule: UNCHANGED from v1 (threshold_v1.json); no threshold is tuned
  on any exam. No frontier-AI output in data, labels, filtering, or selection.

## Pre-training amendment (2026-11-19, BEFORE any v2 training or exam scoring — disclosed)
The off-exam weight fit (`fit_weights_v2.py`, 32,885 non-Dolch-target Birkbeck
pairs; 1,961 Dolch-target pairs excluded by the firewall) gives the rare new
taxonomy families very small raw weights (letter_name 0.0004, affrication
0.00009, th_fv 0.0012 — they are rare in Birkbeck's mostly-older-writer mix,
which is exactly why the published child-taxonomy presence was pre-registered
as a fitting component). Two clarifications, frozen before training:
1. Training-pool sampling uses the fitted weights with a **floor of 0.01 per
   family**, renormalized — every family stays represented in training.
2. The fresh generated exam is **stratified by family** (up to 1 variant per
   family per word, exam seed), not weight-sampled — an exam must cover every
   family for per-family scores to mean anything; it is not a training sample.
3. Exam construction clarification (same pre-training window, disclosed): the
   deterministic generator yields only ~5 variants/word, so a pair-disjoint
   exam cannot be what's left after training sampling. Frozen construction:
   per word per family, variants are split BEFORE training sampling — if a
   family yields ≥2 variants for a word, 1 (exam-seed choice) is reserved for
   the exam; a single-variant family is assigned to exam or training by an
   exam-seed coin flip; held-out families go entirely to the exam. The exam
   remains pair-disjoint from all v2 training data and from v1's examples.

---

# v2 — RESULTS (2026-11-19, local CPU on this VM, 0 Colab units)
Full numbers: `metrics_v2.json`. Scripts: `fit_weights_v2.py`, `gen_v2_data.py`,
`train_v2.py`. Model: `model_v2.npz`, 2,520 bytes (v1: 2,521).

## Weight fit (off-exam, as pre-registered)
Birkbeck amalgamation parsed: 36,133 pairs / 6,136 targets (matches published
counts). After the alpha/single-token filter and the Dolch-target firewall:
32,885 pairs fitted, 1,961 Dolch-target pairs excluded. Category proportions:
other 0.531, generic_drop 0.226, vowel_sub 0.144, doubling 0.037,
sonorant_omit 0.022, transpose 0.020, cluster_reduce 0.016, th_td 0.0029,
th_fv 0.0012, letter_name 0.0004, affrication 0.00009. Full mapping in
`data/family_weights_v2.json`. Note: with ~4–5 candidate variants per word,
the per-word cap (14) seldom binds, so the fitted weights had limited effect
on training composition — recorded plainly.

## Data
Training pool 1,356 rows (split train 994 / val 228 / dev-test 134, seed
20261119); fresh generated exam 261 rows (exam seed 99173), pair-disjoint
from all v2 training data and from v1's examples. Exam family coverage is
skewed by that disjointness (deterministic families' variants were mostly
consumed by v1's development data): transpose 86, generic_drop 68,
vowel_swap 30, cluster_reduce 26, th_fv 21, nasal_liquid_omit 13,
letter_name 6, affrication 6, single_double 5; all other families n=0 on
this exam. v2 dev-test (development numbers only): top-1 0.881 / top-3 0.970.

## Fresh generated exam — matched same-exam scoring (n=261)
| Ranker | top-1 | top-3 |
|---|---|---|
| **v2 product** | **0.720** | **0.935** |
| v1 product | 0.705 | 0.908 |
| B0 Levenshtein | 0.487 | 0.766 |

Per-family top-1 (families with n≥3; v2 / v1 / B0):
affrication (6) 0.667 / 0.667 / 0.667 · cluster_reduce (26) 0.577 / 0.385 / 0.346 ·
generic_drop (68) 0.647 / 0.632 / 0.456 · letter_name (6) 0.167 / 0.167 / 0.000 ·
nasal_liquid_omit (13) 0.846 / 0.846 / 0.462 · single_double (5) 1.000 / 1.000 / 1.000 ·
th_fv (21) 0.286 / 0.238 / 0.333 · transpose (86) 0.837 / 0.872 / 0.430 ·
vowel_swap (30) 1.000 / 1.000 / 0.933.
The v2 gain comes mainly from cluster_reduce (+0.19 over v1, a trained new
family) and th_fv (+0.05); v2 is slightly WORSE on transpose (−0.035).
Held-out families: affrication ties v1; letter_name is poor for both models
(0.167 — single-letter attempts like "r" for "are" carry almost no signal
for a pair ranker). th_fv remains weak for everyone (even B0 beats both
models there): th→f/v attempts are genuinely hard for this feature set,
whose sound form maps th→t.

## Adversarial probes (pre-registered, report-only)
- Homophone collisions (exam rows whose attempt is another vocab word, n=11):
  v2 top-1 0.000, v1 top-1 0.000 — by design: the product exact-match rule
  trusts a correctly spelled attempt, so a collision attempt is always
  "corrected" to itself. Unchanged from v1; context is the known fix.
- Out-of-vocab (65 rule-generated attempts at 18 non-vocab words; unchanged
  v1 confidence rule): acceptance v2 0.215 vs v1 0.185 — v2 rejects 78.5% of
  OOV attempts, slightly worse than v1's 81.5%. Report-only; no gate.

## False-correction
Product self-rank 1.0000 (314/314, exact-match rule) · raw model 0.9745.

## Gates
| Gate | Result |
|---|---|
| 1. Fresh generated exam top-1: v2 ≥ v1 (same exam) | **PASS** — 0.720 ≥ 0.705 |
| 2. False-correction stays 0 (product self-rank = 1.000) | **PASS** — 1.0000 |
| 3. Real exam CHES: v2 ≥ v1 AND v2 ≥ B0 + 0.05 | **NOT EVALUATED — BLOCKED** (standalone CHES file unobtainable legitimately today; see pre-registration acquisition record). Blocked = not passed. |

## Verdict
**NOT KEEPER.** Gates 1–2 pass and the realism lever shows a real, if small,
matched-exam gain (+0.015 top-1, +0.027 top-3 over v1, driven by the new
cluster_reduce family), but gate 3 — the frozen CHES real-exam gate that was
the honesty point of this version — could not be evaluated, so v2 cannot be
declared keeper on this run. **v1 remains the keeper.** No retry of the same
lever today (lineup stop rule). What unblocks it: obtaining the OTA 0643
package with its separate CHES file (OTA handle resolver was returning
HTTP 500 on 2026-11-19; retry on another day, or OTA's download-all-files
route from the item page) and scoring v1 / v2 / B0 on CHES∩Dolch-314
exactly as pre-registered — the exam, gates, and both trained models are
frozen and waiting. Nothing published or pushed. C1b not started.

---

## Views + specialists plan — Phase 1 pointer (2026-11-19)

Phase 1 of `PLAN-VIEWS-SPECIALISTS.md` is DONE (validation split only; the
generated exam and CHES were never opened; no v3 trained). Combined new
views (V4 onset-rime, th-aware sound form, V3 letter-name features, V5
family-posterior features; `features_phase1.py`, base `features.py`
untouched) score **0.9342 validation top-1 vs the v2 control's 0.8333 —
+10.09 pts, clearing the plan's pre-registered +2.0 pt bar**; th_fv
0.20→0.80, cluster_reduce 0.25→0.75 (small n: 5 and 4). V5 alone is the
strongest single view (solo 0.9167). S2 letter-name lookup: 0 validation
triggers (family held out by design), component check passes on the 5
real canonical pairs; false-correction invariant holds (product self-rank
1.0000). Full tables, the V5 generator-consistency caveat, and a
correction to Phase 0's letter-name pair count ('oh' is not in the
vocabulary; 6 real pairs, 5 S2-reachable) are in the plan's Phase 1
results section and `phase1_results.json`. Next step is Phase 2 —
a v3 pre-registration — and the CHES gate (Step 0 of the plan) still
blocks any keeper declaration.

---

## Release note — Phase 1 view probe pushed as DEVELOPMENT update (2026-11-19)

Pushed at Rusty's instruction ("push the winners") as a clearly-labeled
DEVELOPMENT update — **not a release, not a new keeper; v1 remains the
keeper and no v3 has been trained.**

- Repo: https://github.com/RustyDrops/Sound_Out_Speller_Tiny_ML (public), branch `main`
- Commit: `867ff8819888ecac738630057352f48e6bd19390` — "Speller Phase 1 view probe (dev, validation-only): combined views +10.09 pts; v1 remains keeper, CHES gate pending" (parent e6bd6f92; 8 files)
- Contents (staged snapshot in `release_phase1/`): README.md (v1 section intact + new "Phase 1 view probe — DEVELOPMENT (validation only, not a release)" section with the combined 0.9342 vs 0.8333 control, V5 solo 0.9167, th_fv 0.20→0.80 small-n caveat, S2 note with the 'oh'-not-in-vocab correction, V5 generator-consistency caveat, false-correction 1.0000, and the standing CHES keeper gate), PLAN-VIEWS-SPECIALISTS.md (full), PROVENANCE.md (full local copy as of the push), features_phase1.py, phase1_probe.py, phase1_results.json, phase0_diag.py, phase0_results.json.
- Excluded: all datasets and exam files, Birkbeck/CHES data (never shipped), model binaries (no new model exists), caches/logs.
- Verification (read-back via GitHub API): newest commit on main matches; git blob SHAs match local exactly for all 8 pushed files, including phase1_results.json (`e76c3ad9…`) and features_phase1.py (`bee20c2a…`); pre-existing v1 files (model_v1.npz, metrics_v1.json, train_v1.py, demo/, promo/, SPEC.md) confirmed still in tree.

---

## Elephant-Goldfish review of the views + specialists round (2026-11-19)

EG review done — full record in `~/workspace/micro-edge-pilot/EG-REVIEW-2026-11-19.md`
(35 findings: 29 accepted, 6 accepted-with-changes, 0 rejected; Luna cost
$0.006224; feasibility lens run by the elephant after two goldfish
disconnects, logged in the review). Amended next round for this model:
**v3** — pre-register and train FIRST in the restart order (local CPU,
0 units), but only after the L5 weight refit (attempt-overlap exclusion
added to the Dolch-target firewall; the amalgamated Birkbeck file's own
terms recorded in writing; v2's fitted weights are NOT reused). Gates:
fresh generated exam (new seed) v3 >= v1 AND v3 >= v2, false-correction
0, per-family floors, size <= 16 KB, plus the CHES gate verbatim
whenever CHES is obtainable — CHES still blocked means v3 is NOT
KEEPER regardless of generated numbers. A leave-one-family-out
generator-variant diagnostic is included and labelled a robustness
check, not a transfer bound. C1b vocab growth stays held.

---

# v3 — WEIGHT REFIT (EG L5 hardening; done 2026-11-19 BEFORE any v3 training, fitting for v3, or exam scoring)

Per EG-REVIEW-2026-11-19 finding L5 (accepted-with-changes) and its adopted
hardening, mandatory before v3 trains. Script: `fit_weights_v3.py`.
Output: `data/family_weights_v3.json`. **v2's fitted weights are NOT reused
by v3.**

## The amalgamated file's own usage terms — recorded in writing (L5 (ii))
- The local file `data/birkbeck_eval_only/missp.dat` is the `birkbeck` file
  from Roger Mitton's own download page (dcs.bbk.ac.uk/~ROGER/corpora.html,
  re-read 2026-11-19): "an amalgamation of errors taken from the
  native-speaker section (British or American writers) of the Birkbeck
  spelling error corpus... available as separate files with detailed
  documentation from the Oxford Text Archive." The page offers the file for
  download and states **no separate license or reuse terms for the
  amalgamated file itself**; the file carries no license header.
- The underlying corpus is Oxford Text Archive item 0643 (Birkbeck Spelling
  Error Corpus, Mitton 1985); OTA catalogue lists it "Publicly Available"
  and OTA standard terms are research use, no redistribution.
- Project treatment (the stricter of the two governs): the amalgamation is
  used under the OTA terms — EVAL/FITTING ONLY, locally; no Birkbeck/CHES
  data or derived data ships, is published, or is committed to any repo.
  CHES rows are inside the amalgamation but are NOT separately
  identifiable and are never relabelled as CHES.

## Refit rules (L5 (i) + stricter reading, as implemented)
Same deterministic edit-operation classifier and frozen category→family
mapping as `fit_weights_v2.py`. Exclusions, in order:
1. Form filters (as v2): both strings must be `[a-z]+`, target != attempt.
2. **v2 firewall (kept):** target in Dolch-314 → excluded.
3. **NEW attempt-overlap exclusion:** attempt string (normalized =
   lowercased, the v2 fit's handling) appears among the attempt strings of
   ANY Dolch-target row in the file → excluded. Stricter reading adopted:
   the exclusion set is built from every parsed Dolch-target row BEFORE
   any other filter is applied (1,638 distinct attempt strings), because an
   attempt recorded against a Dolch word could constitute exam material
   (the exam is CHES∩Dolch-314 attempts) regardless of how its own row
   would have been filtered. Rationale recorded per the tasking: exclude
   any fitting pair whose attempt string appears in, or could constitute,
   exam material.

## Refit result
- Pairs in file: 36,133 · fitted: **32,279** (v2 fit: 32,885) ·
  skipped by Dolch-target firewall: 1,961 · skipped by the NEW
  attempt-overlap exclusion: **606** · skipped by form filters: 1,287.
- Category proportions barely move (other 0.5334 vs v2 0.531; generic_drop
  0.2246 vs 0.2263; letter_name falls 0.0004 → 0.00009 — most single-letter
  attempts are, unsurprisingly, also recorded against Dolch words). As in
  v2, the 0.01 per-family floor in training sampling limits the practical
  effect; the refit is a provenance hardening, not a performance lever,
  and is reported as such.

---

# v3 — PRE-REGISTRATION (written 2026-11-19 BEFORE any v3 training or exam scoring; refit above precedes it)

Views + specialists plan, Phase 2 (`PLAN-VIEWS-SPECIALISTS.md`), as amended
by EG-REVIEW-2026-11-19 §4 (Speller). Local CPU only, 0 Colab units.
Entered from `V3-PREREGISTRATION-DRAFT-2026-11-19.md` with the draft's two
bracketed choices CONFIRMED (training seed 20261120, exam seed 41729) and
two amendments the draft predated: the generator weights are the **refit
v3 weights** (section above), not v2's; and the G7 generator-variant
diagnostic is specified below.

## The one lever
Feature views ONLY. The ranker's feature set grows from v2's frozen 16
(`features.py`, untouched) by the four Phase 1 groups in
`features_phase1.py`, frozen exactly as Phase 1 left them (EG L6):
- V4 onset-rime extension (3 feats) — Phase 1: +1.32 val top-1 alone.
- TH th-aware sound form (3 feats) — Phase 1: +1.32 alone; th_fv 0.20→0.40.
- V3 letter-name features (3 feats) — Phase 1: +0.00 alone; a guard, not a
  ranking view. Included because the plan froze it in the combined
  configuration that passed the bar; ablated below.
- V5 developmental-family posterior (22 feats) — Phase 1: +9.21 alone
  (solo 0.9167, strongest single view).
Total: 47 features. Plus ONE named deterministic component, not a lever:
- S2 letter-name lookup (~0 bytes, deterministic, NOT learned): ranks the
  parsed word first when an attempt parses cleanly as letter names, the
  parsed word is in the vocabulary, and the attempt is not itself a
  vocabulary word. Vocab-checked coverage is 5 pairs (r→are, y→why, u→you,
  c→see, b→be); 'oh' is not in Dolch-314 so no o→oh pair exists; i→eye is
  excluded by design (exact-match rule). Phase 1 validation effect: zero
  triggers in 228 attempts — no validation number exists for S2 and none is
  claimed; its component check is a smoke check, not an evaluation (L6:
  the canonical pairs informed the lookup, so they are not independent
  evidence).

UNCHANGED from v2: the misspelling generator rules (frozen, L6), the
Birkbeck leakage firewall AS HARDENED by the refit above (weights fitted
only on non-Dolch-target pairs AND with the attempt-overlap exclusion;
CHES never used in training, fitting, filtering, or selection, in any
form), vocabulary (Dolch-314 — C1b vocab growth stays held, EG F7),
classifier recipe (LogisticRegression C=4.0, class_weight balanced, same
negative-sampling and identity-half procedure as v1/v2), the product
exact-match rule, and the confidence rule (`threshold_v1.json`, unchanged;
no threshold tuned on any exam). Training-pool sampling uses the refit
weights with the same 0.01 floor. No learned specialist is part of v3
(S1/S3 rejected on paper in Phase 0; EG F10 keeps them dropped on pools of
this size). No frontier-AI output in data, labels, filtering, or selection.
Seeds: training/split seed **20261120**; exam seed **41729**.

## Honest caveat carried from Phase 1 (frozen into this registration)
V5's features ask "could this candidate have generated this attempt under
family f?" — and the generated attempts were produced by those same
generator families. Part of V5's +9.21 pts is therefore
generator-consistency signal: an upper-bound-flavored number. It transfers
to real child errors only insofar as the published taxonomy (Read 1975;
Treiman 1993; Bear et al.) describes them. The fresh generated exam (new
seed) and CHES — not the validation split — are where that transfer is
judged. Per EG G7, a fresh seed from the same generator tests seed
variation, NOT transfer; the only transfer bound is CHES (blocked).

## Exams (frozen)
- (a) Fresh generated exam: v2 generator construction with exam seed
  41729 (per word per family, variants split BEFORE training sampling;
  single-variant families by exam-seed coin flip; held-out families
  letter_name + affrication entirely to the exam), pair-disjoint from the
  v3 training pool AND excluding every (attempt, target) pair present in
  v1's examples, v2's training pool, or v2's exam. Scored for v1, v2, and
  v3 product rankers — matched same-exam comparison. Per-family top-1
  reported for all three, denominators beside every figure (G13).
- (b) Real exam: CHES filtered to Dolch-314 targets, scored for v1, v3,
  and B0. **Status at pre-registration: BLOCKED** (OTA resolver HTTP 500,
  2026-11-19; amalgamation will NOT be relabelled as CHES). If it remains
  unobtainable, gate 3 is recorded as NOT EVALUATED.
- Adversarial probes (named, report-only), same as v2: (i) homophone /
  collision probe (top-1 0.000 by design under the exact-match rule;
  reported, never promised fixable); (ii) out-of-vocab probe under the
  unchanged v1 confidence rule, reported for v1, v2, v3.
- (c) G7 generator-variant diagnostic (report-only, labelled): the th_fv
  family has two surface realizations (th→f, th→v). A diagnostic model
  with the v3 configuration is trained on the v3 train split MINUS all
  th→v-realization rows, and scored on fresh th→v-only exam rows
  (diagnostic seed 7331, pair-disjoint as in (a)). This is a robustness
  check against rule memorization, NOT a transfer bound (G7).

## Mandatory per-component ablations (report-only; v16 block style)
On the fresh exam: control (base 16), +V4, +TH, +V3, +V5, combined
(views-only = v3 without S2), combined + S2 (= v3 product), and
leave-one-group-out for each of V4/TH/V3/V5. Same configurations also
reported on the v3 validation split. A component whose removal improves
the combined exam top-1 is reported as such — it is NOT quietly dropped;
dropping it is a new version's lever.

## Gates (ALL must pass for v3 to be keeper; frozen)
1. Fresh generated exam (a): v3 product top-1 ≥ v1 product top-1 AND
   v3 product top-1 ≥ v2 product top-1, on the SAME exam (EG G8 — beating
   v2 alone establishes nothing, v2 was never keeper; reporting never
   shortens "passed the generated-exam gate" to "passed the v2 gate").
2. False-correction stays 0: product self-rank rate = 1.000 (314/314),
   with S2 on and off.
3. Real exam (b): v3 CHES top-1 ≥ v1 CHES top-1 AND v3 CHES top-1 ≥ B0
   CHES top-1 + 0.05. (Blocked = not passed; if CHES is still blocked at
   v3 exam time, v3 is NOT KEEPER regardless of generated numbers.)
4. Per-family floors on the fresh exam: th_fv top-1 ≥ 0.286 and
   cluster_reduce top-1 ≥ 0.577 (v2's exam numbers for those families).
5. Size: product model ≤ 16 KB.

## Stop rules
A version failing any gate is rejected, no same-lever retry. Costs: every
step local CPU, 0 Colab units. Nothing is published or pushed without
Rusty's approval of the exact version.

## Pre-training clarifications (2026-11-19, BEFORE any v3 training or exam scoring — disclosed, in the v2 amendment pattern)
A first run of `gen_v3_data.py` under the letter of the construction above
produced a 102-row exam covering only 4 families: for most word/family
combinations the exam-seed's reserved variant was already present in v2's
data (forbidden), and the construction dropped the family for that word
instead of reserving another variant — which would have made gate 4
(th_fv / cluster_reduce floors) unevaluable by construction, the v2
exam-skew problem in worse form. Two clarifications, frozen before
training:
1. **Exam reservation among allowed variants:** per word per family, the
   exam-seed choice is made among the variants whose (attempt, target)
   pair is NOT in the forbidden set (v1 examples ∪ v2 train pool ∪ v2
   exam). If at least one allowed variant exists it is reserved for the
   exam (held-out families: exam only, as before); the remaining variants
   form the training items as in v2's construction. A single-variant
   family whose only variant is forbidden contributes that variant to
   training only. Pair-disjointness and all exclusions are unchanged —
   only the choice set for the reservation is clarified.
2. **G7 diagnostic set:** the diagnostic exam is the subset of the fresh
   generated exam (a) whose family is th_fv AND whose surface realization
   is th→v. (A separate th→v file is impossible: the generator is
   deterministic for this family, so every th→v pair already exists in
   v2's data or the v3 pools.) The diagnostic model is trained on the v3
   train split minus ALL th→v-realization rows and scored on those exam
   rows, alongside v3 product and v2 on the same rows. Disclosed: v2 (and
   v3 product) trained on th→v pairs in their own pools, so for them
   these rows are not realization-unseen; the diagnostic's claim rests on
   the diagnostic-model contrast — can the v3 configuration rank a
   realization it never trained on — which is exactly the robustness
   question G7 posed. Denominators reported (G13).
3. **Exam exclusion set narrowed to true exam material (feasibility):**
   a second run under clarifications 1–2 still produced a 115-row exam
   covering only 4 families — excluding v2's TRAINING pool as well as its
   exam consumes nearly the whole deterministic variant space (v2's own
   results section disclosed the same consumption effect against v1's
   examples alone), leaving th_fv and cluster_reduce at n=0 and gate 4
   unevaluable by construction. Frozen correction: the exam excludes
   every pair in v1's examples and in **v2's exam** (the spent exam —
   reusing its pairs would double-use an exam), and remains pair-disjoint
   from all v3 splits. Pairs that appeared in v2's *training pool* are
   admissible exam material for v3. Disclosed bias: v2 trained on some
   admissible pairs, so on those rows the matched comparison is in-sample
   for v2 and therefore **conservative for v3** under gate 1 (v3 must
   still match or beat v2). This is the last construction clarification;
   the generator now runs once, below.

---

# v3 — RESULTS (2026-11-19, local CPU on this VM, 0 Colab units, ~11 min)
Full numbers: `metrics_v3.json`. Scripts: `fit_weights_v3.py`,
`gen_v3_data.py`, `train_v3.py`. Model: `model_v3.npz`, 3,482 bytes.
Data: training pool 1,484 rows (split train 1,085 / val 245 / dev-test 154,
seed 20261120); fresh generated exam 203 rows (exam seed 41729).

## Fresh generated exam — matched same-exam scoring (n=203)
| Ranker | top-1 | top-3 |
|---|---|---|
| **v3 product (views + S2)** | **0.9310** | **0.9951** |
| v3 views-only (no S2) | 0.9310 | 0.9951 |
| v2 product | 0.7685 | 0.9163 |
| v1 product | 0.7286 | 0.9064 |
| B0 Levenshtein | 0.4675 | — |

Per-family top-1 (v3 / v2, denominators beside figures per G13):
cluster_reduce (20) 0.800 / 0.600 · generic_drop (50) 0.840 / 0.780 ·
nasal_liquid_omit (8) 1.000 / 1.000 · single_double (9) 1.000 / 1.000 ·
th_fv (21) 0.952 / 0.333 · transpose (79) 0.987 / 0.835 ·
vowel_swap (16) 1.000 / 0.938.
Coverage disclosure (same consumption effect v2 recorded, stronger):
only 7 of 20 families have exam rows — the deterministic variant space
was largely consumed by v1's examples and v2's exam (both excluded) plus
the v3 split; the held-out families letter_name and affrication
contributed **0 rows** (every variant was forbidden), so this exam says
nothing about them. Disclosed bias (clarification 3): some exam pairs
appeared in v2's *training* pool, so v2's 0.7685 is partly in-sample and
the gate-1 comparison is conservative for v3. The standing V5 caveat also
applies in full: a fresh seed from the same generator tests seed
variation, not transfer to real child errors — CHES is the only transfer
bound, and it is still blocked.

## Ablations (fresh exam top-1; validation in parentheses)
control base-16 0.7586 (0.8490) · +V4 0.7734 (0.8408) · +TH 0.7685
(0.8694) · +V3 0.7537 (0.8490) · **+V5 0.9310 (0.9224)** · combined
0.9310 (0.9184) · combined−V4 0.9310 · combined−TH 0.9310 (val 0.9224 —
removing TH improves *validation* by +0.40 pt but leaves the exam
unchanged; reported per the pre-registration, NOT dropped — dropping it
is a new version's lever) · combined−V3 0.9310 · combined−V5 0.8030
(0.8490). Reading, plainly: **V5 (family posterior) carries essentially
the entire win**; the other three groups are neutral on this exam.
S2's exam effect is exactly zero — the exam contains no letter_name rows
(see coverage disclosure), so there is no exam number for S2 and none is
claimed; its component check remains a smoke check (Phase 1).

## Probes (report-only)
- Homophone collisions (n=6): v3 / v2 / v1 top-1 all 0.000 — by design
  (exact-match rule), as pre-registered.
- Out-of-vocab (65 attempts, unchanged v1 confidence rule): acceptance
  v3 **0.323** vs v2 0.215 vs v1 0.185 — a named blemish: v3 *accepts
  more* out-of-vocab attempts (its V5-driven margins trip the unchanged
  margin ≥ 1.5 rule more often). Report-only per pre-registration; any
  fix is a confidence-rule version of its own, not a silent retune.

## G7 generator-variant diagnostic (report-only; robustness check, NOT a transfer bound)
Diagnostic model = v3 configuration trained with all th→v-realization
rows removed (7 train rows removed). On the fresh exam's 13 th→v rows:
diagnostic model **1.000 (13/13)**, v3 product 1.000 (trained on th→v —
not realization-unseen, per clarification 2), v2 product 0.4615 (6/13).
The v3 feature configuration ranks a surface realization it never
trained on at ceiling on this small set — evidence against pure rule
memorization by the views, at n=13, labelled as exactly that.

## False-correction
Product self-rank 1.0000 (314/314) with S2 on and off · raw model 0.9841.

## Gates
| Gate | Result |
|---|---|
| 1. Fresh generated exam: v3 ≥ v1 AND v3 ≥ v2 (same exam) | **PASS** — 0.9310 ≥ 0.7286 and ≥ 0.7685 |
| 2. False-correction stays 0 (self-rank = 1.000, S2 on/off) | **PASS** — 1.0000 / 1.0000 |
| 3. Real exam CHES: v3 ≥ v1 AND v3 ≥ B0 + 0.05 | **NOT EVALUATED — BLOCKED** (standalone CHES file unobtainable legitimately; OTA resolver HTTP 500. Blocked = not passed.) |
| 4. Family floors: th_fv ≥ 0.286, cluster_reduce ≥ 0.577 | **PASS** — 0.952 (n=21), 0.800 (n=20) |
| 5. Size ≤ 16 KB | **PASS** — 3,482 B |

## Verdict
**Generated-exam gates PASS (1, 2, 4, 5); KEEPER DEFERRED to CHES.**
Gate 3 — the frozen CHES real-exam gate — could not be evaluated, so v3
cannot be declared keeper on this run regardless of generated-exam
results. **v1 remains the keeper.** v3 ships internally as "candidate —
generated-exam only" (EG F6). What unblocks the verdict: obtaining the
OTA 0643 package with its separate CHES file and scoring v1 / v3 / B0 on
CHES∩Dolch-314 exactly as pre-registered — the exam, gates, and trained
model are frozen and waiting. Nothing published or pushed.

---

# CHES gate retired — owner decision (2026-11-19)

Owner decision (Rusty, 2026-11-19 22:10 CST): drop the CHES real exam and
move on without it. The project proceeds on generated-exam evidence.

- **Background.** The standalone CHES file (one of the per-source files in
  the Oxford Text Archive package for item 0643, the Birkbeck Spelling
  Error Corpus) remained unobtainable across repeated attempts on
  2026-11-19: the OTA handle resolver returned HTTP 500 and retries were
  exhausted (recorded in the v2 pre-registration and the v3 sections
  above). Mitton's own download page offers only the amalgamated file, in
  which CHES rows are not separately identifiable, and relabelling
  amalgamated rows as CHES was never an option.
- **Decision.** Gate 3 (the CHES real-exam gate) is RETIRED — for v2, for
  v3, and for the keeper question generally. It is not marked passed: it
  was never evaluated. It is removed from the gate set by owner decision.
- **Permanent consequence (binding on every public claim).** The
  Sound-Out Speller has NEVER been scored on real children's
  misspellings. Every accuracy figure for every version comes from
  code-generated exams (plus validation splits of generated data). For as
  long as that remains true, every public claim about this model must
  carry the disclosure **"generated-exam only"** — the README's v3
  default-model section carries it verbatim.
- **Birkbeck handling unchanged.** The amalgamated file remains
  eval/fitting-only under the OTA terms, locally; no Birkbeck/CHES data
  or derived data ships, is published, or is committed to any repo.
  Nothing about its handling changes with this retirement.
- **Standing check.** If the standalone CHES file ever becomes
  legitimately obtainable, scoring the frozen v1 / v2 / v3 (and B0) on
  CHES∩Dolch-314 exactly as pre-registered remains a standing, welcome
  check — but it no longer gates anything: not the keeper verdict, not
  the default model.

---

# Verdict amendments under the CHES retirement (2026-11-19)

Amended gate set after the retirement above: v3 gates 1, 2, 4, 5 (gate 3
retired, never evaluated, NOT passed). The original verdict sections
above are NOT rewritten — they stand as the record under the
pre-retirement gate set.

## v3 — KEEPER (under the amended gate set)
v3's verdict is amended from "KEEPER DEFERRED to CHES" to **KEEPER**
under the amended gate set: gate 1 PASS (fresh generated exam 0.9310 ≥
v1 0.7286 and ≥ v2 0.7685, matched same-exam), gate 2 PASS
(false-correction 1.0000), gate 4 PASS (th_fv 0.952 ≥ 0.286;
cluster_reduce 0.800 ≥ 0.577), gate 5 PASS (3,482 B ≤ 16 KB); gate 3
retired by owner decision (section above) — cited here, never restated
as passed. Under Rusty's standing rule (passes its gates + improves →
pushed to GitHub as the DEFAULT model; older models kept in the repo for
comparison), v3 becomes the default model in
RustyDrops/Sound_Out_Speller_Tiny_ML; v1 and v2 stay in the repo for
comparison. The "generated-exam only" disclosure applies to v3 in full:
it has never been scored on real children's misspellings.

## v2 — superseded by v3 the same day; never the default
v2's record is amended as follows (its full pre-registration and results
sections above stand intact): under the amended gate set, v2's evaluable
gates (1 and 2) passed, but v2 is **superseded by v3 on the same day
(2026-11-19)** — v3 beat it on the matched fresh exam (0.9310 vs
0.7685) and on both gated families. v2 was never declared keeper as a
release, was never the default model in the repo, and is retained in
the repo (`model_v2.npz`, `metrics_v2.json`) for comparison only.

---

## Release note — v3 pushed as DEFAULT model (2026-11-19)

Executed under the CHES retirement and verdict amendments above, and
Rusty's standing rule (passes amended gates + improves → GitHub default;
older models kept for comparison).

- Repo: https://github.com/RustyDrops/Sound_Out_Speller_Tiny_ML (public), branch `main`
- Commit: `ca63c7ed1ced289d5d6a525af428b331fd82ed20` — "Speller v3 becomes default model (0.931 generated exam; CHES gate retired by owner decision 2026-11-19, generated-exam-only disclosure in README)" (11 files, staged snapshot in `release_v3/`)
- Contents: README.md (restructured — v3 presented as the DEFAULT model with the generated-exam-only disclosure in its headline section; "Previous models, kept for comparison" covering v1 the former keeper, v2 never-default/superseded, and the condensed Phase 1 dev section), model_v3.npz (default model binary, 3,482 B), model_v2.npz + metrics_v2.json (comparison), metrics_v3.json, features_phase1.py, train_v3.py, gen_v3_data.py, fit_weights_v3.py, PROVENANCE.md (current, incl. the retirement + verdict-amendment sections), PLAN-VIEWS-SPECIALISTS.md.
- Excluded: all datasets and exam files, Birkbeck/CHES data (never shipped), caches/logs. v1 files already in the repo untouched.
- Verification (read-back via GitHub API): newest commit on `main` matches `ca63c7ed…`; git blob SHAs match local exactly for all 11 pushed files, including model_v3.npz (`8a4009e9…`) and README.md (`dd79d0a1…`); pre-existing v1 files (model_v1.npz, metrics_v1.json, train_v1.py, SPEC.md, demo/) confirmed still in tree (38 files total).

---

# C1b — NGSL / New Dolch procurement + frozen 1,000-word vocab (2026-11-20)

Decision: autonomous/c1b-DECISION-2026-11-20.md (C1b approved by Rusty,
decision delegated to Ann, 2026-11-20 14:34 CST). This section records
procurement and pre-registration only — no C1b training has run, v3
remains keeper / GitHub default (ca63c7ed), and no GitHub push was made.

## Licence — re-verified at the source at download time (2026-11-20)
- Source: NGSL / New Dolch List (Browne & Culligan), newgeneralservicelist.com, **CC BY-SA 4.0, commercial allowed** — CONFIRMED on both official pages at download time. The site FAQ states newgeneralservicelist.com is the only official site and the former .org domain is no longer authorised; nothing was taken from .org. Full verbatim evidence + file SHA-256s: `data/ngsl_ndl_source/LICENCE-EVIDENCE-2026-11-20.md`.
- Attribution, verbatim in PROVENANCE, README, and shipped model.json: "Vocabulary includes words from the New General Service List / New Dolch List by Browne & Culligan, licensed CC BY-SA 4.0."
- Share-alike is recorded as applying to the vocabulary list as shipped; no other project data is re-licensed by this.

## Frozen vocab — data/vocab_c1b.json (data/vocab.json NOT overwritten)
Built by build_vocab_c1b.py; audit: autonomous/c1b-vocab-audit.json.
- Step 1: Dolch-314 verbatim, levels unchanged, Dolch wins duplicates → 314.
- Step 2: New Dolch List 1.1 in NDL rank order (NDL_11_stats.csv): +613 new (261 duplicates skipped, 0 normalisation exclusions; o'clock→oclock per apostrophe rule) → 927.
- Step 3: rule fired = NGSL fill — NGSL 1.2 headwords in rank order (NGSL_12_stats.csv): +73 (363 duplicates skipped before target reached) → exactly 1,000 unique normalised words.
- Normalisation: lowercase; apostrophes/hyphens removed; multiword excluded; headword only. All 686 additions get level prior index 2.
- Family weights: v3 family weights reused unchanged (data/family_weights_v3.json) — C1b stays a data-only, one-lever candidate.
- Gates frozen in autonomous/c1b-preregistration.md. CHES gate RETIRED (Rusty, 2026-11-19) — retired, never passed; generated-exam-only disclosure stands.

---

## C1b RESULTS — run on Colab CPU, 2026-11-20 (finalize in train_c1b.py, single run)

**Verdict: C1b PASSES all frozen gates and beats v3 on the expanded exam — C1b becomes the GitHub default, v3 kept for comparison.** Full gate table: autonomous/c1b-VERDICT.md; raw numbers: metrics_c1b.json.

- Training route: Colab CPU exclusively (one bundled zip via GCS staging; runtime deleted after downloads were verified locally — size match + npz/json load checks — and verified gone via list_runtimes). Job record: ~/workspace/colab/speller-c1b/state.json + recover_state.json (the first driver was SIGTERM-killed mid-poll at ~220 s; the runtime was adopted by recover_c1b.py and the in-runtime training completed untouched — training itself ran exactly once). Two earlier driver attempts failed instantly on a bundle-nesting path bug before any training started (0.00 units each).
- Colab units: balance delta over the successful job window 0.17 (account-wide; another chat's T4 assignment was active for part of the window — estimate), rate×elapsed estimate 0.071 (CPU 0.08/hr × ~3,184 s runtime-active — estimate). Pre-registration estimate was ~0.16.
- Fresh generated exam (seed 90210, n=2,403; pair-disjoint from all prior pools and the C1b train pool — 1,557 exam candidates dropped for disjointness, recorded in data/c1b_gen_meta.json): **C1b top-1 0.9434**, B0 0.7711 (margin +0.1723), Dolch-314 slice 0.8941 (n: see metrics), v3 on the same expanded exam 0.0337 (v3 ranks over its Dolch-314 vocab only; targets outside it score 0 — disclosed, and the reason the comparison is structural, not a like-for-like quality drop).
- Per-family C1b top-1 beside headline: th_fv 0.9615 (n=26), cluster_reduce 0.8235 (n=102), generic_drop 0.8400, transpose 0.9922, vowel_swap 0.9294, single_double 1.0000; weakest: affrication 0.1364 (n=22 — held out of training by design, as in v2/v3), ai_e 0.6667 (n=15).
- False-correction: product self-rank 1.0000 over the full 1,000-word vocab (v3 product definition; S2 on == off by construction); raw model self-rank 0.9920, reported for honesty.
- Birkbeck-filtered exam (eval-only, OTA 0643, never shipped, NOT CHES): n=5,499 pairs, C1b 0.4244 ≥ B0 0.3959.
- Size: model_c1b.npz 6,767 B (≤16 KB gate); demo model.json vocab-array encoding 8,958 B, counted and reported.
- CHES gate: RETIRED (Rusty, 2026-11-19) — cited as retired, never passed. Disclosure stands: generated-exam only, never scored on real children's misspellings.
- Attribution: "Vocabulary includes words from the New General Service List / New Dolch List by Browne & Culligan, licensed CC BY-SA 4.0."
- Terms safety: no frontier-model output in data, labels, filtering, or selection; family weights = v3 weights reused unchanged; Birkbeck file eval-only and never shipped.

## Release note — C1b pushed as DEFAULT model (2026-11-20)

Executed under the C1b pre-registration above and Rusty's standing rule
(passes all frozen gates + improves → GitHub default; older models kept).

- Repo: https://github.com/RustyDrops/Sound_Out_Speller_Tiny_ML (public), branch `main`
- Commit: `3d75ca76adec65263375886cbf07263183782449` — "Speller C1b becomes default model (0.9434 fresh generated exam over frozen 1,000-word vocab; all frozen gates pass; CHES gate retired by owner decision 2026-11-19, generated-exam-only disclosure in README; NGSL/New Dolch CC BY-SA 4.0 attribution in README)" (14 files, staged snapshot in `release_c1b/`, parent = v3 default commit `ca63c7ed`)
- Verification (read-back via GitHub API): newest commit on `main` matches `3d75ca76…`; git blob SHAs match local exactly for all 14 pushed files, including model_c1b.npz; pre-existing v1/v3 files confirmed still in tree (49 files total).
- Full verdict: autonomous/c1b-VERDICT.md.

---

# C2-AFFR — affrication enters the training family mix (2026-11-20)

Pre-registration: `autonomous/c2-affr/PREREGISTRATION.md` (frozen before
any C2 training). Verdict: `autonomous/c2-affr/VERDICT.md`.

## The one lever
Training-generator family mix ONLY: affrication (tr→chr, dr→jr) is no
longer held out — it enters training under the standard non-held-out
single-variant rule at the pre-registered 50% rate (exam-seed coin flip
per eligible word; 28 eligible words, 1 deterministic variant each; the
flip sent 18 variants to the fresh exam and 10 to the training pool, all
10 surviving weighted sampling). Letter_name remains held out.
UNCHANGED from C1b: frozen 1,000-word vocab, recipe (LR C=4.0 balanced,
hard-confuser + sampled negatives, identity-half), 47 features, S2,
v3 family weights reused verbatim. Seeds: train 20261301, exam 103101.
Exam construction: pair-disjoint from the C2 train pool for all
families; non-affrication pairs also disjoint from all prior pools
incl. C1b's spent exam; affrication pairs exempt from prior-*exam*
exclusion only (the 28-pair space was fully consumed by prior exams
while held out of all prior training — registered before training).

## Results (run on Colab CPU, 2026-11-20; finalize in train_c2.py, single run)
**Verdict: C2-AFFR PASSES all frozen gates and beats C1b — C2 becomes
the GitHub default, C1b kept for comparison.** Full gate table:
`autonomous/c2-affr/VERDICT.md`; raw numbers: `metrics_c2.json`.

- Fresh generated exam (seed 103101, n=1,754): **C2 top-1 0.9538**,
  B0 0.7623, C1b on the same exam 0.9447 (in-sample-flavoured, see
  preregistration rule 2); Dolch-314 slice 0.9167 (n=72).
- **Affrication 0.8889 (n=18)** — C1b scored 0.1111 and B0 0.5000 on the
  same rows; C1b's own exam had affrication at 0.1364 (n=22). Ten
  training rows closed the gap: an exposure gap, not a feature gap.
- False-correction: product self-rank 1.0000 (S2 on and off, computed
  explicitly); raw model 0.9880.
- Birkbeck-filtered (eval-only, OTA 0643, never shipped, NOT CHES):
  n=5,499, C2 0.4264 ≥ B0 0.3959.
- Size: model_c2.npz 6,767 B (≤16 KB gate); demo vocab array 8,958 B.
- Colab units (estimates, both ways): balance delta 1.69 over the job
  window (account-wide; two other assignments active — overstates C2's
  share); rate × elapsed 0.083 (CPU 0.08/hr × ~3,746 s runtime-active).
  Incident: the first driver was SIGTERM-killed mid-poll;
  recover_c2.py adopted the orphaned runtime and finished the job —
  training ran exactly once.
- CHES gate: RETIRED (Rusty, 2026-11-19) — cited as retired, never
  passed. Disclosure stands: generated-exam only, never scored on real
  children's misspellings.
- Attribution: "Vocabulary includes words from the New General Service List / New Dolch List by Browne & Culligan, licensed CC BY-SA 4.0."
- Terms safety: no frontier-model output in data, labels, filtering, or
  selection; family weights = v3 weights reused unchanged; Birkbeck
  file eval-only and never shipped.
