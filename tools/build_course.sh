#!/usr/bin/env bash
# Build one course end to end: composite every section in parallel, then join.
#   tools/build_course.sh <out-dir> <name>          e.g. tools/build_course.sh build tpp-secure
# Compositing is single-threaded Python per section, so the five run side by side.
set -euo pipefail
OUT=${1:?out dir}; NAME=${2:?course name}
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
pids=()
for seg in $(python3 -c "import json;print(' '.join(s['name'] for s in json.load(open('tools/enhance.json'))['segments']))"); do
  python3 tools/enhance_video.py --manifest tools/enhance.json --out "$OUT" --only "$seg" > "$OUT/$seg.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
tail -n 3 "$OUT"/part-*.log
[ $fail = 0 ] || { echo "a section failed — see $OUT/*.log"; exit 1; }
python3 tools/enhance_video.py --manifest tools/enhance.json --out "$OUT" --join-only --join "$NAME"
