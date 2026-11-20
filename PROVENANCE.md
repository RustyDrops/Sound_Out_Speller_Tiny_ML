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
