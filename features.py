"""Shared feature code for Sound-Out Speller v1.

Deterministic only: a grapheme-simplification "sound form" plus pair features
for (attempt, candidate). No learned components in this file, no external data.
"""
import re

VOWELS = set("aeiou")


def sound_form(w: str) -> str:
    """Approximate how a word sounds, as a simplified spelling.

    This is a similarity code for feature computation, not a pronunciation
    dictionary: consistent, rule-based, and wrong in known ways (e.g. hard-g
    words like 'get'). The learned model weights it against other features.
    """
    s = w.lower()
    s = s.replace("igh", "i")
    s = re.sub(r"^kn", "n", s)
    s = re.sub(r"^wr", "r", s)
    s = re.sub(r"^gn", "n", s)
    s = re.sub(r"^wh", "w", s)
    s = s.replace("ph", "f")
    s = s.replace("ck", "k")
    s = s.replace("th", "t")
    # vowel teams -> canonical single vowel
    s = s.replace("ea", "e").replace("ee", "e")
    s = s.replace("ai", "a").replace("ay", "a")
    s = s.replace("oa", "o").replace("ow", "o").replace("oe", "o")
    s = s.replace("ie", "i")
    s = s.replace("ue", "u").replace("ui", "u")
    # r-controlled vowels -> canonical 'er'
    s = s.replace("ur", "er").replace("ir", "er")
    # soft c / g
    s = re.sub(r"c(?=[eiy])", "s", s)
    s = s.replace("c", "k")
    s = re.sub(r"g(?=[eiy])", "j", s)
    # final y -> i
    s = re.sub(r"y$", "i", s)
    # drop silent final e (after a consonant, word longer than 3)
    if len(s) > 3 and s.endswith("e") and s[-2] not in VOWELS:
        s = s[:-1]
    # collapse double letters
    s = re.sub(r"(.)\1+", r"\1", s)
    return s


def lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _ngrams(s: str, n: int) -> set:
    s2 = "^" + s + "$"
    return {s2[i:i + n] for i in range(len(s2) - n + 1)} if len(s2) >= n else {s2}


def dice(a: str, b: str, n: int) -> float:
    A, B = _ngrams(a, n), _ngrams(b, n)
    if not A or not B:
        return 0.0
    return 2.0 * len(A & B) / (len(A) + len(B))


def _skeleton(w: str, keep_vowels: bool) -> str:
    if keep_vowels:
        return "".join(ch for ch in w if ch in VOWELS)
    return "".join(ch for ch in w if ch not in VOWELS)


def _common_prefix(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def _common_suffix(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(reversed(a), reversed(b)):
        if ca != cb:
            break
        n += 1
    return n


FEATURE_NAMES = [
    "lev_norm", "sound_lev_norm", "sound_equal", "bigram_dice", "trigram_dice",
    "sound_bigram_dice", "vowel_skel_lev", "cons_skel_lev", "same_first",
    "same_last", "prefix_ratio", "suffix_ratio", "len_diff_ratio", "identity",
    "sound_first_equal", "level_prior",
]


def pair_features(attempt: str, cand: str, cand_level_idx: int = 5) -> list:
    a, c = attempt.lower(), cand.lower()
    sa, sc = sound_form(a), sound_form(c)
    mx = max(len(a), len(c), 1)
    ms = max(len(sa), len(sc), 1)
    va, vc = _skeleton(a, True), _skeleton(c, True)
    ka, kc = _skeleton(a, False), _skeleton(c, False)
    return [
        lev(a, c) / mx,
        lev(sa, sc) / ms,
        1.0 if sa == sc else 0.0,
        dice(a, c, 2),
        dice(a, c, 3),
        dice(sa, sc, 2),
        lev(va, vc) / max(len(va), len(vc), 1),
        lev(ka, kc) / max(len(ka), len(kc), 1),
        1.0 if a[:1] == c[:1] else 0.0,
        1.0 if a[-1:] == c[-1:] else 0.0,
        _common_prefix(a, c) / mx,
        _common_suffix(a, c) / mx,
        abs(len(a) - len(c)) / mx,
        1.0 if a == c else 0.0,
        1.0 if sa[:1] == sc[:1] else 0.0,
        1.0 - cand_level_idx / 5.0,
    ]
