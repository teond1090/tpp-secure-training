#!/usr/bin/env python3
"""
check_lipsync.py — measure how far the narration sits from her mouth.

HeyGen's render is not perfectly synchronised: measured against her lips, the
voice arrives about an eighth of a second late, which is past the point where
people stop believing the sound is coming from the speaker. That is worth
correcting, but only with a number we can defend, so this measures it.

HOW
    Her mouth is found per frame rather than sampled from a fixed rectangle —
    she moves her head a great deal in the high-energy take, and a fixed box
    slides off her face and measures cheekbone instead. The silhouette against
    the black backdrop gives the top of her head and its width every frame, and
    the mouth sits at a fixed fraction of head size below that.

    Mouth openness (the spread of luma inside the lip box: a closed mouth is a
    flat band of lipstick, an open one has a dark gap and bright teeth) is then
    cross-correlated against the narration's loudness. The lag at the peak is
    the offset.

CALIBRATION
    Shifting a known amount of audio against the picture moves the reported lag
    by exactly that much, in both directions:

        audio 200 ms earlier   expected +5 frames   measured +5
        audio unchanged        expected  0          measured  0
        audio 200 ms later     expected -5 frames   measured -5

    So the absolute number can be trusted, not just the sign.

USAGE
    python3 tools/check_lipsync.py media/part-1.webm
    python3 tools/check_lipsync.py build/tpp-secure.mp4 --start 30 --seconds 120
"""

import argparse, subprocess, sys

import numpy as np

SW, SH = 960, 540         # half resolution: plenty to find a head, four times cheaper
KEY = 8                   # luma above this is her, not the backdrop
MAX_LAG = 25              # +/- one second of search


def ffmpeg_bin():
    from shutil import which
    return which("ffmpeg") or __import__("imageio_ffmpeg").get_ffmpeg_exe()


def mouth_openness(ff, src, fps, start, dur):
    """Per-frame spread of luma inside a mouth box tracked off her head."""
    p = subprocess.Popen(
        [ff, "-v", "error", "-ss", str(start), "-t", str(dur), "-i", src,
         "-vf", f"scale={SW}:{SH},format=gray", "-r", str(fps),
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        stdout=subprocess.PIPE, bufsize=SW * SH * 4)
    n, out = SW * SH, []
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        g = np.frombuffer(b, np.uint8).reshape(SH, SW)
        mask = g > KEY
        rows = np.where(mask.any(1))[0]
        if not len(rows):
            out.append(0.0); continue
        r0 = rows[0]
        cols = np.where(mask[r0:r0 + 25].any(0))[0]
        if len(cols) < 10:
            out.append(0.0); continue
        cx, hw = (cols[0] + cols[-1]) / 2.0, (cols[-1] - cols[0]) / 2.0
        y0, y1 = max(int(r0 + 2.25 * hw), 0), min(int(r0 + 2.95 * hw), SH)
        x0, x1 = max(int(cx - 0.50 * hw), 0), min(int(cx + 0.50 * hw), SW)
        out.append(float(g[y0:y1, x0:x1].std()) if y1 - y0 > 3 and x1 - x0 > 3 else 0.0)
    p.stdout.close(); p.wait()
    return np.array(out)


def loudness(ff, src, fps, start, dur):
    a = subprocess.run(
        [ff, "-v", "error", "-ss", str(start), "-t", str(dur), "-i", src,
         "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
        capture_output=True, check=True).stdout
    s = np.frombuffer(a, dtype="<i2").astype(np.float32) / 32768
    hop = int(16000 / fps); m = len(s) // hop
    return np.sqrt((s[:m * hop].reshape(m, hop) ** 2).mean(1) + 1e-12)


def correlate(a, b, max_lag=MAX_LAG):
    k = min(len(a), len(b))
    a = (a[:k] - a[:k].mean()) / (a[:k].std() + 1e-9)
    b = (b[:k] - b[:k].mean()) / (b[:k].std() + 1e-9)
    out = []
    for lag in range(-max_lag, max_lag + 1):
        x, y = (a[lag:], b[:k - lag] if lag else b) if lag >= 0 else (a[:k + lag], b[-lag:])
        j = min(len(x), len(y))
        out.append(float((x[:j] * y[:j]).mean()))
    return np.array(out)


def measure(ff, src, fps=25.0, start=8.0, seconds=110.0):
    c = correlate(mouth_openness(ff, src, fps, start, seconds),
                  loudness(ff, src, fps, start, seconds))
    lag = int(c.argmax()) - MAX_LAG
    return lag, float(c.max()), c


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="+")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--start", type=float, default=8.0)
    ap.add_argument("--seconds", type=float, default=110.0)
    ap.add_argument("--tolerance", type=int, default=1,
                    help="frames of offset to accept before reporting a failure")
    args = ap.parse_args()

    ff = ffmpeg_bin()
    bad = 0
    for v in args.video:
        lag, peak, c = measure(ff, v, args.fps, args.start, args.seconds)
        ms = lag / args.fps * 1000
        ok = abs(lag) <= args.tolerance
        bad += not ok
        print(f"{v}: {lag:+d} frames ({ms:+.0f} ms, "
              f"{'mouth behind the voice' if lag > 0 else 'voice behind the mouth'}), "
              f"peak {peak:.2f}  {'ok' if ok else 'OUT OF SYNC'}")
        print("    " + " ".join(f"{l:+d}:{c[l + MAX_LAG]:.2f}" for l in range(-8, 9)))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
