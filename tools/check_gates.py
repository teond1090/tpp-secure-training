#!/usr/bin/env python3
"""
check_gates.py — prove the presenter finishes speaking before every quiz.

The knowledge checks are time-gated: the player pauses the course at each
`gate` in the joined video's .json and puts the question up. If a gate lands
while she is still mid-sentence the course feels broken, so this measures the
finished file rather than trusting the arithmetic that placed the gates.

For each gate it finds the last moment the narration is above the speech floor
and reports the silence between that word and the gate. Anything under MIN is
a failure and the script exits non-zero, so a bad build cannot be published.

USAGE
    python3 tools/check_gates.py media/enhanced/tpp-secure.mp4
    python3 tools/check_gates.py build/tpp-secure.mp4 --min 2.0
"""

import argparse, json, os, subprocess, sys

import numpy as np

SPEECH_FLOOR_DB = -45     # below this is room tone and encoder noise, not words
MIN_SILENCE = 1.5         # seconds she must be silent for before a gate fires
WINDOW = 14.0             # how far back to look for her last word


def ffmpeg_bin():
    from shutil import which
    return which("ffmpeg") or __import__("imageio_ffmpeg").get_ffmpeg_exe()


def envelope(ff, src, start, dur, hop_ms=10):
    """10 ms RMS envelope in dBFS, plus the timestamp of each step."""
    start = max(start, 0.0)
    raw = subprocess.run(
        [ff, "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", src,
         "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
        capture_output=True, check=True).stdout
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
    hop = 16000 * hop_ms // 1000
    n = len(a) // hop
    if not n:
        return np.array([]), np.array([])
    rms = np.sqrt((a[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
    return start + np.arange(n) * (hop_ms / 1000), 20 * np.log10(rms + 1e-9)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="the joined course mp4; its .json holds the gates")
    ap.add_argument("--min", type=float, default=MIN_SILENCE,
                    help=f"seconds of silence required before a gate (default {MIN_SILENCE})")
    args = ap.parse_args()

    meta_path = os.path.splitext(args.video)[0] + ".json"
    if not os.path.exists(meta_path):
        sys.exit(f"no gate metadata beside {args.video} — expected {meta_path}")
    meta = json.load(open(meta_path))
    gates = [s for s in meta["sections"] if s.get("gate") is not None]
    if not gates:
        sys.exit(f"{meta_path} lists no gates")

    ff = ffmpeg_bin()
    bad = 0
    for i, sec in enumerate(gates, 1):
        g = sec["gate"]
        t, db = envelope(ff, args.video, g - WINDOW, WINDOW)
        spoken = np.where(db > SPEECH_FLOOR_DB)[0]
        if not len(spoken):
            print(f"  gate {i} at {g:7.2f}s: silent for the whole {WINDOW:.0f}s window")
            continue
        last = t[spoken[-1]]
        gap = g - last
        ok = gap >= args.min
        bad += not ok
        print(f"  gate {i} at {g:7.2f}s: last word {last:7.2f}s -> "
              f"{gap:5.2f}s of silence  {'ok' if ok else 'TOO SOON'}")

    print(f"{len(gates)} gates, {bad} firing before she finishes")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
