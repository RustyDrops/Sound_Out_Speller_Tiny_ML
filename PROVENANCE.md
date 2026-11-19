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

## Known limitations
See SPEC.md "Honest limits": no -ed pattern words in Dolch, homophones need context,
generated errors approximate — but do not prove performance on — real children's spelling.
