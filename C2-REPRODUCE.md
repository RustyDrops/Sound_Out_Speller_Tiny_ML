# Reproducing C2-AFFR

C2-AFFR was trained on Google Colab (CPU) from a single bundled zip:
`gen_c2_data.py` then `train_c2.py`, exactly as pushed here, with the
project's `features.py`, `features_phase1.py`, `gen_v1_data.py`,
`gen_v2_data.py`, `model_c1b.npz` (same-exam comparison ranker), and
these data files: `data/vocab_c1b.json` (pushed here under `data/`),
`data/vocab.json` (Dolch-314 slice), `data/family_weights_v3.json` (v3
weights, reused unchanged), the prior-pool files (examples_v1, exam_v2,
examples_v2, exam_v3, examples_v3, exam_c1b — used only for the
pair-disjointness check), and `data/birkbeck_eval_only/missp.dat`
(eval-only, OTA terms — never shipped, not included here).

One bundle adaptation, disclosed (same as C1b): in the executed bundle,
the copy of `train_v1.py`'s `load_vocab()` read `data/vocab_c1b.json`
instead of `data/vocab.json`, with the `c1b_additions` level at prior
index 2. That is the only change to that file; the repo's own
`train_v1.py` (v1's script) is untouched.

Seeds: train 20261301, exam 103101. Outputs: `model_c2.npz`,
`metrics_c2.json`. Job record (staging handle, balances, download
verification, runtime deletion, and the SIGTERM/recovery incident) is
in the project workspace under `~/workspace/colab/speller-c2-affr/`
(state.json, recover_state.json).
