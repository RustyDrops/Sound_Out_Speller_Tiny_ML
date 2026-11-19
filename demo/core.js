// Sound-Out Speller v1 — core logic (JS port of features.py + predict_v1.py).
// Pure functions; expects a global MODEL = {coef, intercept, vocab, levelIdx, marginThreshold}.
// This exact file is inlined into demo/index.html by build_demo.py and is also
// exercised directly under node by parity_test.js.

function soundForm(w) {
  let s = String(w).toLowerCase();
  s = s.replace(/igh/g, "i");
  s = s.replace(/^kn/, "n");
  s = s.replace(/^wr/, "r");
  s = s.replace(/^gn/, "n");
  s = s.replace(/^wh/, "w");
  s = s.replace(/ph/g, "f");
  s = s.replace(/ck/g, "k");
  s = s.replace(/th/g, "t");
  s = s.replace(/ea/g, "e").replace(/ee/g, "e");
  s = s.replace(/ai/g, "a").replace(/ay/g, "a");
  s = s.replace(/oa/g, "o").replace(/ow/g, "o").replace(/oe/g, "o");
  s = s.replace(/ie/g, "i");
  s = s.replace(/ue/g, "u").replace(/ui/g, "u");
  s = s.replace(/ur/g, "er").replace(/ir/g, "er");
  s = s.replace(/c(?=[eiy])/g, "s");
  s = s.replace(/c/g, "k");
  s = s.replace(/g(?=[eiy])/g, "j");
  s = s.replace(/y$/, "i");
  if (s.length > 3 && s.endsWith("e") && !"aeiou".includes(s[s.length - 2])) {
    s = s.slice(0, -1);
  }
  s = s.replace(/(.)\1+/g, "$1");
  return s;
}

function lev(a, b) {
  if (a === b) return 0;
  if (!a) return b.length;
  if (!b) return a.length;
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) {
      cur.push(Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] !== b[j - 1] ? 1 : 0)));
    }
    prev = cur;
  }
  return prev[b.length];
}

function ngramSet(s, n) {
  const s2 = "^" + s + "$";
  const out = new Set();
  if (s2.length < n) { out.add(s2); return out; }
  for (let i = 0; i <= s2.length - n; i++) out.add(s2.slice(i, i + n));
  return out;
}

function dice(a, b, n) {
  const A = ngramSet(a, n), B = ngramSet(b, n);
  if (!A.size || !B.size) return 0;
  let inter = 0;
  for (const g of A) if (B.has(g)) inter++;
  return (2 * inter) / (A.size + B.size);
}

function skeleton(w, keepVowels) {
  let out = "";
  for (const ch of w) {
    const isV = "aeiou".includes(ch);
    if (isV === keepVowels) out += ch;
  }
  return out;
}

function commonPrefix(a, b) {
  let n = 0;
  while (n < a.length && n < b.length && a[n] === b[n]) n++;
  return n;
}

function commonSuffix(a, b) {
  let n = 0;
  while (n < a.length && n < b.length && a[a.length - 1 - n] === b[b.length - 1 - n]) n++;
  return n;
}

function pairFeatures(attempt, cand, candLevelIdx) {
  const a = String(attempt).toLowerCase(), c = String(cand).toLowerCase();
  const sa = soundForm(a), sc = soundForm(c);
  const mx = Math.max(a.length, c.length, 1);
  const ms = Math.max(sa.length, sc.length, 1);
  const va = skeleton(a, true), vc = skeleton(c, true);
  const ka = skeleton(a, false), kc = skeleton(c, false);
  return [
    lev(a, c) / mx,
    lev(sa, sc) / ms,
    sa === sc ? 1 : 0,
    dice(a, c, 2),
    dice(a, c, 3),
    dice(sa, sc, 2),
    lev(va, vc) / Math.max(va.length, vc.length, 1),
    lev(ka, kc) / Math.max(ka.length, kc.length, 1),
    a[0] === c[0] ? 1 : 0,
    a[a.length - 1] === c[c.length - 1] ? 1 : 0,
    commonPrefix(a, c) / mx,
    commonSuffix(a, c) / mx,
    Math.abs(a.length - c.length) / mx,
    a === c ? 1 : 0,
    sa[0] === sc[0] ? 1 : 0,
    1 - candLevelIdx / 5,
  ];
}

// Rank the full vocabulary (MODEL.vocab + any extra custom words) for an attempt.
// Returns {ranked: [{word, score}], exact, margin, top}.
function suggest(attempt, extraWords) {
  const a = String(attempt).toLowerCase().trim();
  const vocab = MODEL.vocab.slice();
  const levels = MODEL.levelIdx.slice();
  if (extraWords) {
    for (const w of extraWords) {
      if (!vocab.includes(w)) { vocab.push(w); levels.push(2); }
    }
  }
  const scored = vocab.map((c, i) => {
    const f = pairFeatures(a, c, levels[i]);
    let s = MODEL.intercept;
    for (let j = 0; j < f.length; j++) s += f[j] * MODEL.coef[j];
    return { word: c, score: s };
  });
  scored.sort((x, y) => y.score - x.score);
  const exact = vocab.includes(a);
  let margin;
  if (exact) {
    const mine = scored.find((r) => r.word === a);
    const rest = scored.filter((r) => r.word !== a);
    margin = mine.score - rest[0].score;
    const ranked = [mine].concat(rest);
    return { ranked, exact, margin, top: a };
  }
  margin = scored[0].score - scored[1].score;
  return { ranked: scored, exact, margin, top: scored[0].word };
}

// Confidence rule calibrated in calibrate_threshold2.py:
// accept a suggestion if the top candidate sounds identical (sound distance 0),
// is one edit away, or the model's margin clears the calibrated threshold.
function isAccepted(attempt, result) {
  if (result.exact) return true;
  const a = String(attempt).toLowerCase().trim();
  const top = result.top;
  if (lev(soundForm(a), soundForm(top)) === 0) return true;
  if (lev(a, top) <= 1) return true;
  return result.margin >= MODEL.marginThreshold;
}

// Deterministic phonics hint for (attempt, target): first matching pattern wins.
function hintFor(attempt, target) {
  const a = String(attempt).toLowerCase(), t = String(target).toLowerCase();
  if (a === t) return "";
  if (t.length > 3 && t.endsWith("e") && a === t.slice(0, -1))
    return "This word ends with a silent e: " + t + ".";
  if (t.startsWith("wh") && a.startsWith("w") && !a.startsWith("wh"))
    return "This word starts with wh: " + t + ".";
  const dbl = t.match(/(.)\1/);
  if (dbl && a === t.replace(dbl[0], dbl[1]))
    return "This word has a double letter: " + dbl[0] + " in " + t + ".";
  if (t.includes("th") && (a.includes("t") || a.includes("d")) && lev(a, t) <= 2)
    return "This word has th in it: " + t + ". Tongue between your teeth!";
  if (t.includes("ck") && a.replace(/k$/, "c") === t || (t.includes("ck") && a.includes("k") && !a.includes("ck") && lev(a, t) <= 2))
    return "That k sound is spelled ck here: " + t + ".";
  if (t.endsWith("y") && a.endsWith("i") && a.slice(0, -1) === t.slice(0, -1))
    return "This word ends with y: " + t + ".";
  if (soundForm(a) === soundForm(t))
    return "You heard every sound! This word just spells them a tricky way: " + t + ".";
  if (a.length === t.length && lev(a, t) === 2) {
    const sa = a.split("").sort().join(""), st = t.split("").sort().join("");
    if (sa === st) return "All the letters are there — two of them swapped places.";
  }
  if (lev(a, t) === 1 && a.length < t.length)
    return "You are one letter away — a letter is missing.";
  if (lev(a, t) === 1 && a.length > t.length)
    return "You are one letter away — one extra letter snuck in.";
  if (lev(a, t) === 1)
    return "You are one letter away — check each letter against " + t + ".";
  return "You heard the sounds — look closely at the middle of " + t + ".";
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { soundForm, lev, pairFeatures, suggest, isAccepted, hintFor };
}
