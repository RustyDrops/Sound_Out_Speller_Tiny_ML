"""Export model_v1.npz to demo/model.json for the browser demo."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
data = np.load(HERE / "model_v1.npz")
thr = json.loads((HERE / "threshold_v1.json").read_text())
out = {
    "coef": [round(float(x), 6) for x in data["coef"]],
    "intercept": round(float(data["intercept"][0]), 6),
    "vocab": [str(w) for w in data["vocab"]],
    "levelIdx": [int(i) for i in data["level_idx"]],
    "marginThreshold": thr["margin"],
}
(HERE / "demo").mkdir(exist_ok=True)
(HERE / "demo" / "model.json").write_text(json.dumps(out))
print(f"wrote demo/model.json ({len(out['vocab'])} words, margin {thr['margin']})")
