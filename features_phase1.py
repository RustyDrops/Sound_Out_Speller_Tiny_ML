"""Sound-Out Speller — Phase 1 view features (PLAN-VIEWS-SPECIALISTS.md).

Extends the frozen v1/v2 feature set WITHOUT modifying features.py: the
first 16 outputs of pair_features_ext() are exactly features.pair_features(),
so the v2 control is reproducible by column selection. New groups, in the
plan's implementation order:

  V4 onset-rime extension      onset_lev, rime_lev, onset_len_diff
  TH th-aware sound form       thsound_lev_norm, thsound_equal,
                               thsound_bigram_dice  (sound_form_th keeps
                               'th' as its own symbol instead of erasing
                               it to 't'; weighted Levenshtein gives
                               th<->t/d/f/v substitutions cost 0.5)
  V3 letter-name               ln_parses, ln_target_match, ln_len
  V5 family posterior          fam_<family> for all 20 generator families
                               (1/(1 + cheapest family-script cost)) plus
                               best_family_is_th, best_family_is_cluster

Deterministic only, no learned components, no external data. The letter-name
parser and the S2 deterministic lookup live here too (S2_LETTER_NAME_MAP /
parse_letter_name / s2_lookup_rank).
"""
import re

from features import FEATURE_NAMES as BASE_FEATURE_NAMES
from features import VOWELS, dice, lev, sound_form
from gen_v1_data import FAMILIES as V1_FAMILIES
from gen_v2_data import NEW_FAMILIES

# ---------------------------------------------------------------- V4: onset-rime

def _onset_rime(w: str):
    """Onset = leading consonant cluster (possibly empty); rime = the rest."""
    i = 0
    while i < len(w) and w[i] not in VOWELS:
        i += 1
    return w[:i], w[i:]


# ------------------------------------------------------- TH: th-aware sound form

TH_SYMBOL = "þ"  # stands in for the 'th' digraph inside sound_form_th


def sound_form_th(w: str) -> str:
    """sound_form() with 'th' protected as its own symbol (TH_SYMBOL)
    instead of collapsed to 't'. Every other rule mirrors features.sound_form
    in the same order, so the two forms differ ONLY in the th treatment."""
    s = w.lower().replace("th", TH_SYMBOL)
    s = s.replace("igh", "i")
    s = re.sub(r"^kn", "n", s)
    s = re.sub(r"^wr", "r", s)
    s = re.sub(r"^gn", "n", s)
    s = re.sub(r"^wh", "w", s)
    s = s.replace("ph", "f")
    s = s.replace("ck", "k")
    # (no th->t step: that erasure is what this view exists to undo)
    s = s.replace("ea", "e").replace("ee", "e")
    s = s.replace("ai", "a").replace("ay", "a")
    s = s.replace("oa", "o").replace("ow", "o").replace("oe", "o")
    s = s.replace("ie", "i")
    s = s.replace("ue", "u").replace("ui", "u")
    s = s.replace("ur", "er").replace("ir", "er")
    s = re.sub(r"c(?=[eiy])", "s", s)
    s = s.replace("c", "k")
    s = re.sub(r"g(?=[eiy])", "j", s)
    s = re.sub(r"y$", "i", s)
    if len(s) > 3 and s.endswith("e") and s[-2] not in VOWELS:
        s = s[:-1]
    s = re.sub(r"(.)\1+", r"\1", s)
    return s


def _th_sub_cost(x: str, y: str) -> float:
    if x == y:
        return 0.0
    # developmental th substitutions (th_td and th_fv families) cost half
    if (x == TH_SYMBOL and y in "tdfv") or (y == TH_SYMBOL and x in "tdfv"):
        return 0.5
    return 1.0


def lev_th(a: str, b: str) -> float:
    """Levenshtein with th-aware substitution costs (ins/del cost 1)."""
    if a == b:
        return 0.0
    if not a:
        return float(len(b))
    if not b:
        return float(len(a))
    prev = [float(j) for j in range(len(b) + 1)]
    for i, ca in enumerate(a, 1):
        cur = [float(i)]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1.0, cur[j - 1] + 1.0,
                           prev[j - 1] + _th_sub_cost(ca, cb)))
        prev = cur
    return prev[-1]


# ------------------------------------------------------------- V3: letter-name

# Letters whose spoken name is itself a Dolch vocabulary word — the exact
# set the v2 generator's letter_name family produces (gen_v2_data.py).
LETTER_NAME_MAP = {"r": "are", "y": "why", "u": "you", "c": "see",
                   "i": "eye", "b": "be", "o": "oh"}


def parse_letter_name(attempt: str, word_set) -> str | None:
    """Parse the whole attempt as a sequence of letter names. Succeeds iff
    every character is a letter whose name is a vocabulary word AND the
    concatenation of the names is itself a vocabulary word. (Length-1 is
    the only form the generator produces; the general rule reduces to the
    canonical map there.)"""
    a = attempt.lower()
    if not a or any(ch not in LETTER_NAME_MAP for ch in a):
        return None
    parsed = "".join(LETTER_NAME_MAP[ch] for ch in a)
    return parsed if parsed in word_set else None


def s2_lookup_rank(attempt: str, ranked: list, word_set) -> list:
    """S2 deterministic letter-name lookup (product-level rule, ablatable).

    Fires only when the attempt parses cleanly as letter names, the parsed
    word is in the vocabulary, and the attempt is NOT itself a vocabulary
    word — a vocab-word attempt (e.g. 'i') is claimed by the product
    exact-match rule first, by design. Action: parsed word ranked first."""
    a = attempt.lower()
    if a in word_set:
        return ranked
    parsed = parse_letter_name(a, word_set)
    if parsed is None:
        return ranked
    return [parsed] + [w for w in ranked if w != parsed]


# ------------------------------------------------------- V5: family posterior

def _single_double_all(w):
    out = []
    for i in range(1, len(w)):
        if w[i] not in "aeiou" and w[i - 1] in "aeiou":
            out.append(w[:i] + w[i] + w[i:])
    return out


def _vowel_swap_all(w):
    swaps = {"a": "e", "e": "i", "i": "e", "o": "u", "u": "o"}
    return [w[:i] + swaps[ch] + w[i + 1:]
            for i, ch in enumerate(w) if ch in swaps and 0 < i < len(w)]


def _transpose_all(w):
    if len(w) < 3:
        return []
    return [w[:i] + w[i + 1] + w[i] + w[i + 2:] for i in range(len(w) - 1)]


def _generic_drop_all(w):
    if len(w) < 3:
        return []
    return [w[:i] + w[i + 1:] for i in range(1, len(w))]


# All-variant (deterministic) versions of the generator's RNG families, so
# the posterior sees every output the family can produce, not one sample.
_ALL_VARIANT = {
    "single_double": _single_double_all,
    "vowel_swap": _vowel_swap_all,
    "transpose": _transpose_all,
    "generic_drop": _generic_drop_all,
}

# Family order is frozen: v1 dict order, then the v2 new families.
FAMILY_ORDER = list(V1_FAMILIES) + list(NEW_FAMILIES)
_FAMILY_FN = {}
for _name in FAMILY_ORDER:
    if _name in _ALL_VARIANT:
        _FAMILY_FN[_name] = _ALL_VARIANT[_name]
    elif _name in NEW_FAMILIES:
        _FAMILY_FN[_name] = NEW_FAMILIES[_name]
    else:
        _FAMILY_FN[_name] = V1_FAMILIES[_name]

_variant_cache: dict = {}


def _family_variants(word: str) -> dict:
    """family -> all attempts that family can generate FROM `word`
    (the generator direction: candidate-as-target -> attempt). Cached."""
    if word not in _variant_cache:
        out = {}
        for name in FAMILY_ORDER:
            vs = _FAMILY_FN[name](word)
            out[name] = [v for v in dict.fromkeys(vs) if v and v != word]
        _variant_cache[word] = out
    return _variant_cache[word]


def _family_costs(attempt: str, cand: str):
    """Per family: cheapest script cost of explaining attempt as cand with
    one slip. 0 if the family generates the attempt from the candidate
    exactly; otherwise the residual edit distance to the family's closest
    output; families that cannot apply to the candidate fall back to the
    plain attempt<->candidate distance (unexplained, not free)."""
    base = lev(attempt, cand)
    variants = _family_variants(cand)
    costs = []
    for name in FAMILY_ORDER:
        vs = variants[name]
        costs.append(min((lev(attempt, v) for v in vs), default=base))
    return costs


# ------------------------------------------------------------------ assembly

V4_NAMES = ["onset_lev", "rime_lev", "onset_len_diff"]
TH_NAMES = ["thsound_lev_norm", "thsound_equal", "thsound_bigram_dice"]
V3_NAMES = ["ln_parses", "ln_target_match", "ln_len"]
V5_NAMES = [f"fam_{n}" for n in FAMILY_ORDER] + ["best_family_is_th",
                                                 "best_family_is_cluster"]
FEATURE_NAMES_EXT = BASE_FEATURE_NAMES + V4_NAMES + TH_NAMES + V3_NAMES + V5_NAMES

# The letter-name features need the vocabulary (frozen Dolch-314) to decide
# whether a parse lands on a real word; loaded once from the same source
# every other script uses.
from train_v1 import load_vocab as _load_vocab  # noqa: E402

_WORDS, _LVL = _load_vocab()
_WORD_SET = set(_WORDS)


def pair_features_ext(attempt: str, cand: str, cand_level_idx: int = 5) -> list:
    from features import pair_features as _base_pair_features
    a, c = attempt.lower(), cand.lower()
    feats = _base_pair_features(a, c, cand_level_idx)

    # V4 onset-rime
    oa, ra = _onset_rime(a)
    oc, rc = _onset_rime(c)
    feats += [
        lev(oa, oc) / max(len(oa), len(oc), 1),
        lev(ra, rc) / max(len(ra), len(rc), 1),
        abs(len(oa) - len(oc)) / max(len(a), len(c), 1),
    ]

    # TH th-aware sound form
    ta, tc = sound_form_th(a), sound_form_th(c)
    feats += [
        lev_th(ta, tc) / max(len(ta), len(tc), 1),
        1.0 if ta == tc else 0.0,
        dice(ta, tc, 2),
    ]

    # V3 letter-name
    parsed = parse_letter_name(a, _WORD_SET)
    feats += [
        1.0 if parsed is not None else 0.0,
        1.0 if parsed is not None and parsed == c else 0.0,
        float(len(a)) if parsed is not None else 0.0,
    ]

    # V5 family posterior
    costs = _family_costs(a, c)
    feats += [1.0 / (1.0 + cost) for cost in costs]
    best = FAMILY_ORDER[min(range(len(costs)), key=lambda i: costs[i])]
    feats += [
        1.0 if best in ("th_fv", "th_td") else 0.0,
        1.0 if best == "cluster_reduce" else 0.0,
    ]
    return feats
