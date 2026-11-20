# Speller C1b — Decision (delegated by Rusty, made by Ann, 2026-11-20 14:34 CST)

Rusty, verbatim: "You approve and make he decision for speller"
So: **C1b is APPROVED. Procurement approved. The decisions below are frozen for pre-registration.**

## Selection rule (blocker 2) — child-first, frequency-filled, exactly 1,000
Rationale: this is a children's sound-out speller. A pure NGSL top-1,000 is an adult general-service list — frequency-right but age-wrong as the primary source. The New Dolch list is the same project's modernised children's list, so it leads; NGSL fills by proven frequency.
1. Keep all 314 Dolch words verbatim, with their existing levels. Dolch wins every duplicate.
2. Add New Dolch List words not already in Dolch-314, in New Dolch order.
3. Fill to **exactly 1,000 unique normalised words** with NGSL 1.2 headwords in NGSL rank order, skipping duplicates.
4. If steps 1–2 already reach ≥1,000, step 3 adds nothing and the list is cut at 1,000 in that order (record which rule fired).

## Licence mechanics (blocker 3)
- Source: NGSL / New Dolch List (Browne & Culligan), newgeneralservicelist.com, **CC BY-SA 4.0, commercial allowed** — re-verify at the source at download time and record the verification in PROVENANCE.
- Attribution, verbatim in PROVENANCE, README, and shipped model.json: "Vocabulary includes words from the New General Service List / New Dolch List by Browne & Culligan, licensed CC BY-SA 4.0."
- Share-alike is recorded as applying to the vocabulary list as shipped; no other project data is re-licensed by this.

## Level prior (blocker 4)
- All non-Dolch additions get the middle level prior (level 2) — the existing in-repo precedent for Dolch nouns. One rule, no invented rank bands.

## Normalisation / dedup (blocker 5)
- Follow Dolch precedent: lowercase; apostrophes removed (don't→dont); hyphens removed (good-bye→goodbye); multiword entries excluded; headword/lemma form only, no separate inflected forms; Dolch level wins on overlap.

## Family weights (blocker 6)
- **Reuse v3 family weights unchanged.** Refitting with an expanded exclusion set would be a second lever; C1b stays data-only, one lever.

## Birkbeck exam (blocker 7)
- Filter Birkbeck (eval-only, OTA terms, never shipped, not CHES) to the expanded 1,000 targets; report C1b and B0 (plain Levenshtein) on the same filtered set. Gate: C1b ≥ B0.

## Gates — frozen from c1b-prep.md proposed shape, with these numbers
1. Fresh generated exam (new seed, pair-disjoint): top-1 ≥ B0 + 0.10, AND Dolch-314 slice top-1 ≥ 0.881 (v3's 0.931 minus 5 pts non-regression floor for a 3× vocab expansion).
2. False-correction = 0: self-rank 1.000 over the full expanded vocab, S2 on and off.
3. Per-family floors: th_fv ≥ 0.286, cluster_reduce ≥ 0.577, per-family scores beside headline.
4. Birkbeck-filtered: C1b ≥ B0 on same set.
5. Size ≤ 16 KB all-in, shipped vocab accounting stated at pre-registration.
6. CHES gate: RETIRED (Rusty, 2026-11-19) — cited as retired, never passed. Disclosure stands: generated-exam only, never scored on real children's misspellings (Birkbeck is eval-only, not CHES).

## Training route — updated
- Rusty's 2026-11-20 rule supersedes the prep note's local-CPU estimate: **Colab CPU exclusively**, est. ~0.16 units, one bundled zip, runtime deleted after verified downloads.
- Pass all gates + beat v3 on the expanded exam → GitHub default via PAT, v3 kept for comparison. Fail any gate = rejected, v3 stays default.
