"""Splice scratch-iter003/rescreen_block.json into library/candidates.json as a
new top-level key, textually (no reformat of existing content), then validate.
"""
import json

cand = r"D:\claude\Quant_works\hft-autofactor\library\candidates.json"
blk = r"D:\claude\Quant_works\hft-autofactor\scratch-iter003\rescreen_block.json"

text = open(cand, encoding="utf-8").read()
block = open(blk, encoding="utf-8").read().rstrip("\n")

assert "eval_v2_rescreen_2026_08_05" not in text, "already inserted, abort"

# block alone must be a valid object member
wrap = json.loads("{\n" + block + "\n}")
assert len(wrap["eval_v2_rescreen_2026_08_05"]["verdicts"]) == 18

stripped = text.rstrip()
anchor = "\n  ]\n}"
assert stripped.endswith(anchor), repr(stripped[-80:])

new = stripped[: -len(anchor)] + "\n  ],\n" + block + "\n}"
data = json.loads(new)  # full-file validity check
assert "eval_v2_rescreen_2026_08_05" in data
assert len(data["eval_v2_rescreen_2026_08_05"]["verdicts"]) == 18

suffix = "\n" if text.endswith("\n") else ""
with open(cand, "w", encoding="utf-8", newline="") as f:
    f.write(new + suffix)

print("inserted OK")
print("top-level keys:", list(data.keys()))
print("verdicts:", len(data["eval_v2_rescreen_2026_08_05"]["verdicts"]))
