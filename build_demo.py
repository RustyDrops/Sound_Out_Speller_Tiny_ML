"""Build demo/index.html: inline model.json + core.js + level word lists
into template.html so the demo is one self-contained file (no network)."""
import json
from pathlib import Path

HERE = Path(__file__).parent
tpl = (HERE / "demo" / "template.html").read_text()
model = (HERE / "demo" / "model.json").read_text()
core = (HERE / "demo" / "core.js").read_text()
vocab = json.loads((HERE / "data" / "vocab.json").read_text())
levels = {k: v for k, v in vocab["levels"].items()}

html = tpl.replace("__MODEL_JSON__", model)
html = html.replace("__CORE_JS__", core)
html = html.replace("__LEVEL_WORDS__", json.dumps(levels))
out = HERE / "demo" / "index.html"
out.write_text(html)
print(f"wrote {out} ({len(html)} bytes)")
