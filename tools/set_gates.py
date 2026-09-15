#!/usr/bin/env python3
"""Set the player's knowledge-check gates from the join's section table.

    python3 tools/set_gates.py media/enhanced/tpp-secure.json index.html

Each card's mid-point becomes a gate, as a fraction of the finished video,
so the player pauses on the "Knowledge Check" card rather than mid-sentence.
"""
import json, re, sys
meta = json.load(open(sys.argv[1])); html_path = sys.argv[2]
total = meta["seconds"]
gates = [round(s["gate"] / total, 5) for s in meta["sections"] if s["kind"] == "bumper"]
s = open(html_path).read()
found = re.findall(r"\{ at:([0-9.]+), title:", s)
if len(found) != len(gates):
    sys.exit(f"{len(found)} quizzes in the page but {len(gates)} cards in the video")
for old, new in zip(found, gates):
    s = s.replace(f"{{ at:{old}, title:", f"{{ at:{new}, title:", 1)
open(html_path, "w").write(s)
print("gates:", gates, "->", [round(g * total, 1) for g in gates], "s")
