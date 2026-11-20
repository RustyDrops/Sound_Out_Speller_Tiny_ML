# Sound-Out Speller — Views + Specialists Training Plan

Written 2026-11-19. Modeled on Write It Right's current approach: look at the
data several different ways (views), and add small specialist double-checkers
only where a measured failure justifies one. **Phase 0 diagnostics are done
(below); no candidate version has been trained.** Local CPU only, 0 Colab
units. Keeper is **v1** (v2 not keeper — its CHES real-exam gate is blocked,
not passed; see PROVENANCE).

## Step 0 — the blocker that gates everything downstream (stated plainly)

The CHES real exam is still unobtained: the Oxford Text Archive handle
resolver for item 0643 returned HTTP 500 on 2026-11-19, and Mitton's own
download offers only the amalgamated Birkbeck file, in which CHES rows are
**not** separately identifiable (never relabel them). Until CHES lands:

- v2's verdict stays NOT KEEPER (gates 1–2 passed, gate 3 not evaluated).
- **C1b vocab growth stays held** (it was sequenced after the C1 verdict).
- Any candidate from this plan (v3) **cannot be declared keeper either** —
  its gates include the CHES gate exactly as pre-registered for v2
  (candidate ≥ v1 CHES top-1 AND ≥ B0 CHES top-1 + 0.05), evaluated whenever
  CHES becomes available.
- Retry path: OTA resolver on another day, or the item page's
  download-all-files route. v1/v2/B0 and the gates are frozen and waiting;
  no retraining is needed to score them.

Everything else in this plan proceeds on **generated validation data only**,
in parallel with the CHES retry. The fresh generated exam (`exam_v2.jsonl`)
is a spent, frozen exam: it is never used for view selection, specialist
design, or threshold tuning in this plan.

## The views — six ways to look at a (attempt, candidate) pair

Current state: `features.py` computes 16 pair features, which already form
four implicit views (V1, V2, V4, V6 below). V3 and V5 are new. Each view gets
a **solo probe**: a logistic ranker trained on the v2 TRAIN split using only
that view's features (v2 recipe otherwise: C=4.0, balanced, same negative
sampling + identity-half), scored on the v2 VALIDATION split only. A view
that cannot beat ~0.65 solo on validation is not carrying signal worth
fusing; a view whose solo top-3 is high but top-1 low is a re-ranking view,
not a ranking view.

| # | View | What it sees | Features |
|---|---|---|---|
| V1 | **Phonetic / sound** | How the pair sounds under the code-built `sound_form` (th→t, ph→f, vowel teams collapsed) | existing: `sound_lev_norm`, `sound_equal`, `sound_bigram_dice`, `sound_first_equal` |
| V2 | **Grapheme / edit-operation** | The raw letter strings: edit distance and n-gram overlap | existing: `lev_norm`, `bigram_dice`, `trigram_dice`; Phase 1 adds explicit edit-script features (first differing position, substitution pair identity, insertion/deletion position) |
| V3 | **Letter-name** (NEW) | The attempt parsed as letter *names* (are→R, you→U, see→C): a single-letter or name-sequence attempt is a different kind of signal than a phonetic attempt | new: `ln_parses` (deterministic parse succeeds), `ln_target_match` (parsed letter-name word == candidate), `ln_len` |
| V4 | **Skeleton / onset-rime** | Vowel skeleton and consonant skeleton separately; Phase 1 adds a true onset-rime split (onset = initial consonant cluster, rime = the rest) | existing: `vowel_skel_lev`, `cons_skel_lev`; new: `onset_lev`, `rime_lev`, `onset_len_diff` |
| V5 | **Developmental-family** (NEW) | Which known error family (th_fv, cluster_reduce, generic_drop, …) best explains attempt→candidate, via the cheapest family-tagged edit script under the generator rules | new: one feature per major family = 1/(1 + family-script cost), plus `best_family_is_th`, `best_family_is_cluster` indicators |
| V6 | **Word-shape / length + prior** | Edges, prefix/suffix, length, identity, and the Dolch level prior | existing: `same_first`, `same_last`, `prefix_ratio`, `suffix_ratio`, `len_diff_ratio`, `identity`, `level_prior` |

## The specialists — small double-checkers, each with a named failure

A specialist is consulted only when its trigger fires; it may only re-order
the base ranker's **top-3** (it cannot promote a word from rank 4+ — Phase 0
shows why that limit matters). Total deployed size budget: **≤ 16 KB all-in**
(base ranker is 2,521 B; each learned specialist ≤ 4 KB; deterministic
components ~0 B). The SPEC ceiling is 2 MB; the 16 KB budget is the plan's
own discipline, in the spirit of the 2.5 KB original.

- **S1 — th-family re-ranker.** Named failure: `th_fv` (exam 0.286, and plain
  Levenshtein B0 beats the model there, 0.333). Trigger: attempt contains
  f/v/th AND ≥2 of the top-3 candidates contain th/f/v. Training data (when
  built): code-generated th_td + th_fv rows only, family weights behind the
  existing Birkbeck firewall (non-Dolch targets only; no CHES contact).
- **S2 — letter-name lookup (deterministic, NOT learned).** Named failure:
  `letter_name` (0.167 for both models — single-letter attempts carry almost
  no pair signal). Trigger: attempt parses cleanly as a letter-name sequence
  (Phase 0 parser) AND the parsed word is in the vocabulary AND the attempt
  is not itself a vocabulary word. Action: rank the parsed word first.
- **S3 — cluster/onset re-ranker.** Named failure: `cluster_reduce`
  (v1 exam 0.385 → v2 0.577, still weak). Trigger: attempt opens with a
  single consonant + vowel AND ≥1 top-3 candidate opens with a 2-consonant
  cluster containing that letter. Training data: code-generated
  cluster_reduce + nasal_liquid_omit rows, firewall weights as S1.
- **Explicitly NOT a specialist: homophone collisions.** Exam top-1 is
  0.000 **by design** — the product exact-match rule trusts a correctly
  spelled attempt, and a collision attempt *is* a correctly spelled word.
  No re-ranker can fix this; it needs sentence context the product does not
  have. This plan does not promise a fix and assigns no work to it.

## HEADROOM RULE (mandatory, frozen for this plan — the WIR v7 Path A lesson)

Write It Right's image cascade died on train-split arithmetic before any
training: a *perfect* specialist could gain only +0.38 pts against a +0.46
gate, with a breakable-to-addressable ratio of ~14:1. No specialist in this
plan is built until it passes the same kind of arithmetic **on validation**:

1. **Addressable share:** ≥ 60% of the specialist's *triggered errors* have
   the correct word in the base ranker's top-3.
2. **Oracle gain:** fixing every addressable triggered error gains
   ≥ **+2.0 points** of overall validation top-1.
3. **Breakage exposure:** breakable (triggered and already correct) ÷
   addressable ≤ **3:1** — a realistic specialist that is wrong sometimes
   must still net positive.
Fail any leg → the specialist is **rejected on paper**, and the diagnostic
signal is redirected (usually into a view, where it can also reach errors
whose correct word sits below rank 3). Small-n caveat: validation families
here are tiny (th_fv n=5, cluster_reduce n=4), which is exactly why legs 2–3
are stated in overall-validation points and counts, not family percentages.

## Sequencing

- **Step 0:** CHES retry (above). Gates v2's verdict, C1b, and any keeper
  call from this plan.
- **Phase 0 (DONE, below):** headroom per specialist, solo probes of the
  four existing views, letter-name parse check — validation only.
- **Phase 1:** build the two genuinely new views (V3 letter-name features,
  V5 family-posterior features) and the V4 onset-rime extension + V2
  edit-script extension; solo-probe each on train/val; then one combined
  probe (all views) on validation. No exam contact. If combined views do
  not beat the all-features control (0.833 val top-1) by ≥ +2 pts on
  validation, the view lever is reported as weak and the plan stops here.
- **Phase 2 (only for components that pass):** pre-register **v3** in
  PROVENANCE before training. One lever: the new views added to the ranker,
  plus the S2 deterministic lookup if Phase 1 confirms it is safe — with a
  mandatory per-component ablation (each view group / the lookup on-off),
  in the Scam Shield v16 block style. Any learned specialist that somehow
  passes the headroom rule on a larger validation pool joins the same
  pre-registration or waits for its own version — never bundled silently.
- **v3 gates (frozen shape, finalized at pre-registration):** fresh
  generated exam (new seed) matched scoring — v3 ≥ v1 AND v3 ≥ v2 on the
  same exam; false-correction stays 0 (product self-rank 1.000); per-family
  floors on th_fv and cluster_reduce vs v2's exam numbers; size ≤ 16 KB;
  **and the CHES gate exactly as pre-registered for v2** (v3 ≥ v1 and
  ≥ B0 + 0.05 on CHES∩Dolch-314) whenever CHES is available. If CHES is
  still blocked at v3 exam time, v3 is NOT KEEPER regardless of generated
  numbers — the v2 honesty rule.
- **Stop rules:** a version failing any gate is rejected, no same-lever
  retry; a specialist failing the headroom rule is not re-triggered with a
  tweaked trigger on the same validation split (that is tuning on the
  measuring instrument).
- **Costs:** every step is local CPU, seconds to ~2 minutes (Phase 0's full
  run: ~2 min, most of it the solo-probe refits). 0 Colab units throughout.

---

## Phase 0 results (2026-11-19, validation only — `phase0_diag.py`, `phase0_results.json`)

Split replication is exact: v2's split (seed 20261119) reproduced as
train 994 / val 228 / test 134, and the loaded v2 product ranker scores
**0.8333 top-1 on validation — identical to `metrics_v2.json`**, so the
diagnostics below are measuring the real v2.

### (a) Headroom — where v2's validation errors actually sit

Validation errors: 38 of 228. Rank of the correct word when the ranker errs:
**rank 2 in 29 cases, rank 3 in 1, ranks 4–5 in 5, ranks 6–10 in 2, beyond
10 in 1.** So 30 of 38 errors (79%) are top-3-addressable *in principle* —
but they are mostly **generic rank-2 near-misses**, not family-concentrated:
the single largest addressable pool is generic_drop (7 errors, 5 in top-3).
No named family owns the error mass. That fact shapes every verdict below.

| Specialist | Triggered | Breakable | Addressable (err, correct in top-3) | Oracle gain | Verdict vs headroom rule |
|---|---|---|---|---|---|
| S1 th re-ranker | 30 | 26 | 2 (of 4 triggered errors = 50%) | **+0.88 pts** | **REJECTED on paper** — fails all three legs (50% < 60%; +0.88 < +2.0; ratio 13:1). Same shape as WIR Path A (14:1). |
| S3 cluster re-ranker | 15 | 11 | 3 (of 4 triggered errors = 75%) | **+1.32 pts** | **REJECTED on paper** — passes addressable share (75%) but fails oracle gain (+1.32 < +2.0) and ratio (3.7:1 > 3:1). Narrow miss, recorded as such. |
| S2 letter-name lookup | — | — | not measurable on validation | — | **Deterministic exception** — see (c). Not a learned specialist; no headroom gate applies, but its ceiling is bounded and stated. |

Why S1 fails structurally, not just arithmetically: of v2's 4 validation
th_fv errors (n=5, top-1 0.20), only 1 has the correct word in the top-3;
the others sit at ranks 4–5 and 6–10. A top-3 re-ranker *cannot reach them
at all*. th attempts are hard for this feature set because `sound_form`
maps th→t, collapsing the very contrast the family turns on. **Redirect:**
the th signal belongs in a *view* — a th-aware sound form (keep th distinct;
add an f/v↔th equivalence class) inside V1/V5 — where it can move
candidates into the top-3, not just shuffle them. S3's redirect is the same
shape: **all 3 of its addressable errors sit at rank 2**, so an onset
feature in the main ranker (V4 onset-rime extension) attacks them without
a trigger's breakage exposure.

### (b) View solo probes — validation top-1 / top-3 (v2 recipe, one feature group at a time)

| View (existing feature group) | Solo top-1 | Solo top-3 |
|---|---|---|
| V1 phonetic (`sound_*`, 4 feats) | 0.662 | 0.829 |
| V2 grapheme/edit (lev + dice, 3 feats) | 0.719 | 0.921 |
| V4 skeleton (vowel/cons skeletons, 2 feats) | **0.750** | 0.895 |
| V6 edge/shape (edges + length, 5 feats) | 0.746 | **0.934** |
| identity+prior (2 feats) | 0.004 | 0.013 |
| **All 16 features (control, retrained in-script)** | **0.833** | **0.965** |

Read: the control reproduces v2 exactly (0.8333), so the solo numbers are
comparable. Two honest findings: (1) the **phonetic view is the weakest
real view** despite being the model's namesake idea — consistent with the
th diagnosis (its sound form erases th and merges f/ph); (2) the
identity+prior "view" is degenerate alone (0.004) — it is a tie-breaker and
false-correction guard, not a view, and Phase 1 should not count it as one.
The skeleton view (2 features, 0.750 solo) is the most efficient signal in
the current model — support for investing in its V4 onset-rime extension.
V3 and V5 have no existing features, so they have no solo numbers yet;
their probes are Phase 1 work, gated by the ≥ +2 pt combined rule above.

### (c) Letter-name parse check

Deterministic parser (letter → its name-word, from the generator's own
definitions in `gen_v2_data.py`; rankers were **not** scored on these, and
the exam file was not opened): **7 of 7 canonical letter-name attempts
parse cleanly** (r→are, y→why, u→you, c→see, i→eye, b→be, o→oh), each
mapping to exactly one vocabulary word. One structural exception, stated
plainly: the attempt **"i" is itself a vocabulary word**, so the product
exact-match rule claims it before any lookup could run — the same
by-design collision class as the homophones. S2's usable coverage is
therefore **6 of 7** canonical pairs at ~0 bytes. Its ceiling is small by
construction (letter_name's fitted Birkbeck weight was 0.0004; 6 of 261
exam rows) — worth including in v3 as a named, ablated deterministic
component, never worth a learned model. Note also: `letter_name` has
**zero validation rows** (held out of the v2 training pool by design), so
no validation headroom number exists for it; anyone quoting one is
measuring on the exam.

### Phase 0 bottom line

- **No learned specialist passes the headroom rule.** Both are rejected on
  paper, cheaply, before any building — the WIR Path A outcome, reached
  the same honest way.
- **The first implementable step is Phase 1's view work**, in this order:
  (1) V4 onset-rime extension (skeleton is the strongest cheap view; it
  owns the cluster rank-2 errors), (2) a th-aware sound form in V1/V5
  (owns the th errors that sit below rank 3), (3) V3 letter-name features
  + the S2 deterministic lookup (bounded, ~free), (4) V5 family-posterior
  features. Combined-probe bar: ≥ +2 pts over 0.833 validation top-1, or
  the view lever is reported weak and the plan stops before any v3.

---

## Phase 1 results (2026-11-19, validation only — `phase1_probe.py`, `phase1_results.json`, features in `features_phase1.py`)

Implemented in the plan's order, as new feature groups appended to the frozen
16 (features.py untouched; the control below is the base-16 columns of the
extended matrix, trained with the v2 recipe and Phase 0's exact negative
sampling). **Sanity gate passed: the control reproduces v2's validation
numbers exactly (0.8333 / 0.9649)** — every delta below is measured against
the real v2. Validation n=228 throughout; th_fv n=5 and cluster_reduce n=4,
so family numbers move in whole-child steps and are reported with that
caveat.

### Configurations — validation top-1 / top-3 (product ranker)

| Configuration | top-1 | top-3 | Δ top-1 vs control | th_fv (n=5) | cluster (n=4) |
|---|---|---|---|---|---|
| Control (base 16 = v2) | 0.8333 | 0.9649 | — | 0.20 | 0.25 |
| + V4 onset-rime (3 feats) | 0.8465 | 0.9693 | +1.32 | 0.20 | **0.75** |
| + TH th-aware sound form (3 feats) | 0.8465 | 0.9693 | +1.32 | **0.40** | 0.25 |
| + V3 letter-name features (3 feats) | 0.8333 | 0.9649 | +0.00 | 0.20 | 0.25 |
| + V5 family posterior (22 feats) | 0.9254 | 0.9956 | **+9.21** | **0.80** | **0.75** |
| **Combined (47 feats)** | **0.9342** | **0.9956** | **+10.09** | **0.80** | **0.75** |
| Combined + S2 lookup | 0.9342 | 0.9956 | +10.09 | 0.80 | 0.75 |

### View-only solos (plan's solo-probe protocol, new groups alone)

| View | Solo top-1 | Solo top-3 | Read |
|---|---|---|---|
| V4 onset-rime | 0.5965 | 0.7632 | Weak alone, useful fused; owns cluster (solo cluster 1.00) |
| TH th-aware | 0.6623 | 0.8070 | ≈ the old phonetic view solo (0.662) but keeps the th contrast |
| V3 letter-name | 0.0000 | 0.0088 | Degenerate alone, as expected: its features never fire in the training pool (family held out) — it is a guard, not a ranking view |
| V5 family posterior | **0.9167** | 0.9912 | **Beats the 16-feature control solo.** The single strongest view in the model |

### The bar — PASS

Combined 0.9342 vs control 0.8333 = **+10.09 pts ≥ the +2.0 pt bar.** The
view lever is NOT weak; Phase 2 (pre-registering v3) is unlocked on the
validation evidence. Per-family, combined vs control: th_fv 0.20→0.80,
cluster_reduce 0.25→0.75, transpose 0.82→0.97, vowel_swap 0.78→0.89,
y_i 0.81→1.00, c_k 0.88→1.00, silent_e_drop 0.83→1.00,
nasal_liquid_omit 0.86→1.00; generic_drop remains the weakest real family
(0.53→0.60).

**Honest caveat on V5's size of win:** the family-posterior features ask
"could this candidate have generated this attempt under family f?" — and
the validation attempts were produced by those same generator families.
Part of the +9 pts is therefore generator-consistency signal: an
upper-bound-flavored number. It should transfer to real child errors only
insofar as the published taxonomy (Read/Treiman/Bear) describes them. The
fresh generated exam at v3 time (new seed) and CHES — not this validation
split — are where that transfer gets judged.

### S2 letter-name lookup — ablation and component check

- **Validation effect: exactly zero, stated plainly.** S2 triggers on
  **0 of 228** unique validation attempts (letter_name has n=0 validation
  rows by design), and combined+S2 is bit-identical to combined on
  validation. There is no validation number for S2 and none is claimed.
- **Component smoke check** (the canonical attempts; a component check,
  not an evaluation set): under the combined ranker alone, the targets of
  r/y/u/c/b sit at raw ranks 58 / 17 / 12 / 177 / 4; with S2 on, all five
  rank first. The lookup does exactly its named job at ~0 bytes.
- **Correction to Phase 0's count:** Phase 0 counted 7 canonical pairs
  from the generator's static dict. The vocab-checked count is **6 real
  pairs**: 'oh' is **not in the Dolch-314 vocabulary**, so the generator's
  oh→o entry can never fire and no such pair exists anywhere. Usable S2
  coverage is **5 pairs** (r, y, u, c, b); 'i'→eye stays excluded by
  design (the attempt 'i' is itself a vocabulary word, claimed by the
  exact-match rule). S2's ceiling is even smaller than Phase 0 stated —
  it remains a free deterministic component for v3, never a lever.

### False-correction invariant — holds

Combined product ranker self-rank on the full vocabulary: **1.0000**
(314/314), and 1.0000 with S2 on. Raw combined model alone: 0.9841
(better than v2's raw 0.9745, still below the gate — the exact-match rule
remains load-bearing, as designed).

### Scope note and what remains

This run implemented the four ordered steps from Phase 0's bottom line
(V4 onset-rime, th-aware sound form, V3 + S2, V5 family posterior). The
V2 edit-script extension mentioned in the views table was not built — the
combined bar passed without it; it stays available as a future addition.
**No v3 was trained and nothing here is a candidate.** Phase 2 still
requires: (1) a v3 pre-registration in PROVENANCE before training, with
per-component ablations and the frozen gate shape from this plan —
including the CHES gate verbatim; (2) **CHES itself (Step 0) — v3, like
v2, cannot be declared keeper until the real exam lands**, whatever the
generated numbers say.
