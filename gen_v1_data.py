"""Sound-Out Speller v1 — deterministic misspelling generator.

Applies seeded, rule-based phonics-error families to the Dolch vocabulary in
data/vocab.json and writes data/examples_v1.jsonl with fields:
  {attempt, target, family}
Correct spellings are NOT included here; the false-correction check in
train_v1.py feeds vocabulary words directly.

No model of any kind is involved: every example is produced by the explicit
string rules below (see PROVENANCE.md).
"""
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).parent
SEED = 42
MAX_VARIANTS_PER_WORD = 14

HELD_OUT_FAMILIES = {"wh_w", "er_ur"}  # excluded from training, exam only (SPEC.md)


def _sub_all(word, pat, repl):
    out, seen = [], set()
    for m in re.finditer(pat, word):
        v = word[:m.start()] + repl(m) + word[m.end():]
        if v and v != word and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def fam_silent_e_drop(w):
    return [w[:-1]] if len(w) > 3 and w.endswith("e") and w[-2] not in "aeiou" else []


def fam_vowel_team_swap(w):
    out = []
    swaps = [("ea", "ee"), ("ea", "e"), ("ee", "ea"), ("ai", "ay"), ("ay", "ai"),
             ("oa", "o"), ("ow", "ou"), ("ou", "ow"), ("ie", "i"), ("oo", "u")]
    for a, b in swaps:
        out += _sub_all(w, re.escape(a), lambda m, b=b: b)
    return out


def fam_ai_e(w):
    return _sub_all(w, "ai", lambda m: "e") + _sub_all(w, "ai", lambda m: "a")


def fam_double_single(w):
    return _sub_all(w, r"(.)\1", lambda m: m.group(1))


def fam_single_double(w, rng):
    out = []
    for i in range(1, len(w)):
        if w[i] not in "aeiou" and w[i - 1] in "aeiou":
            out.append(w[:i] + w[i] + w[i:])
    return [rng.choice(out)] if out else []


def fam_ck_k(w):
    return _sub_all(w, "ck", lambda m: "k") + _sub_all(w, "ck", lambda m: "c")


def fam_c_k(w):
    out = []
    if w.startswith("c"):
        out.append("k" + w[1:])
    out += _sub_all(w, "k", lambda m: "c")
    return out


def fam_wh_w(w):
    return ["w" + w[2:]] if w.startswith("wh") else []


def fam_th_td(w):
    return _sub_all(w, "th", lambda m: "t") + _sub_all(w, "th", lambda m: "d")


def fam_vowel_swap(w, rng):
    swaps = {"a": "e", "e": "i", "i": "e", "o": "u", "u": "o"}
    out = []
    for i, ch in enumerate(w):
        if ch in swaps and 0 < i < len(w):
            out.append(w[:i] + swaps[ch] + w[i + 1:])
    return [rng.choice(out)] if out else []


def fam_unstressed_drop(w):
    if len(w) < 5:
        return []
    out = []
    for i in range(1, len(w) - 1):
        if w[i] in "aeiou":
            out.append(w[:i] + w[i + 1:])
    return out[:2]


def fam_transpose(w, rng):
    if len(w) < 3:
        return []
    i = rng.randrange(len(w) - 1)
    return [w[:i] + w[i + 1] + w[i] + w[i + 2:]]


def fam_y_i(w):
    out = []
    if w.endswith("y"):
        out.append(w[:-1] + "i")
    out += _sub_all(w, "i", lambda m: "y")
    return out


def fam_er_ur(w):
    return _sub_all(w, "er", lambda m: "ur") + _sub_all(w, "ur", lambda m: "er")


def fam_generic_drop(w, rng):
    if len(w) < 3:
        return []
    i = rng.randrange(1, len(w))
    return [w[:i] + w[i + 1:]]


FAMILIES = {
    "silent_e_drop": fam_silent_e_drop,
    "vowel_team_swap": fam_vowel_team_swap,
    "ai_e": fam_ai_e,
    "double_single": fam_double_single,
    "single_double": fam_single_double,
    "ck_k": fam_ck_k,
    "c_k": fam_c_k,
    "wh_w": fam_wh_w,
    "th_td": fam_th_td,
    "vowel_swap": fam_vowel_swap,
    "unstressed_drop": fam_unstressed_drop,
    "transpose": fam_transpose,
    "y_i": fam_y_i,
    "er_ur": fam_er_ur,
    "generic_drop": fam_generic_drop,
}

RNG_FAMILIES = {"single_double", "vowel_swap", "transpose", "generic_drop"}


def main():
    vocab_doc = json.loads((HERE / "data" / "vocab.json").read_text())
    words = []
    for level in ["pre_primer", "primer", "first", "second", "third", "nouns"]:
        words += vocab_doc["levels"][level]
    words = sorted(set(words))
    rng = random.Random(SEED)
    rows = []
    for w in words:
        variants = {}  # attempt -> family (first family wins, deterministic order)
        for fam_name, fn in FAMILIES.items():
            outs = fn(w, rng) if fam_name in RNG_FAMILIES else fn(w)
            for v in outs:
                if v != w and v not in variants:
                    variants[v] = fam_name
        items = sorted(variants.items())
        if len(items) > MAX_VARIANTS_PER_WORD:
            items = sorted(rng.sample(items, MAX_VARIANTS_PER_WORD))
        for attempt, fam in items:
            rows.append({"attempt": attempt, "target": w, "family": fam})
    rng.shuffle(rows)
    out_path = HERE / "data" / "examples_v1.jsonl"
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    fam_counts = Counter(r["family"] for r in rows)
    collisions = sum(1 for r in rows if r["attempt"] in set(words) and r["attempt"] != r["target"])
    print(f"vocab words: {len(words)}")
    print(f"examples: {len(rows)}  (collisions with another vocab word: {collisions})")
    for fam, n in sorted(fam_counts.items()):
        tag = " [HELD OUT]" if fam in HELD_OUT_FAMILIES else ""
        print(f"  {fam}: {n}{tag}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
