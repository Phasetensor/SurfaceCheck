import json
from pathlib import Path

totals = json.loads(Path("coverage.json").read_text())["totals"]
for covered, total, required in (
    ("covered_lines", "num_statements", 95),
    ("covered_branches", "num_branches", 90),
):
    value = 100 * totals[covered] / totals[total]
    print(f"{covered} {value:.2f}%")
    if value < required:
        raise SystemExit(1)
