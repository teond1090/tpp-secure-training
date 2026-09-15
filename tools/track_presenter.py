#!/usr/bin/env python3
"""A virtual camera for the presenter card.

The render is her on black, full height. A fixed square crop shows her
large, but the high-energy delivery throws her hands past its edges — in
one frame in eight with a 1080px crop. A wider fixed crop keeps the hands
and makes her small all the time.

This does what a camera operator does: hold the tight frame while she
talks, pull back only while a gesture is wide, ease back in afterwards.
Because the backdrop and the card are both black, a wider crop simply adds
invisible headroom above her; she stays anchored at the bottom.

Pass 1 measures, per frame, how far anything bright reaches from centre —
face, hands, shirt. Only the bright parts matter: a black sleeve crossing a
black card edge is invisible. Pass 2 crops each frame to the smoothed
width, resizes it to the card, and hands the frames to ffmpeg.

    python3 tools/track_presenter.py media/part-2.webm card.mp4
    python3 tools/track_presenter.py media/part-2.webm card.mp4 --start 55 --dur 25
"""
import argparse, json, re, subprocess, sys
import numpy as np
from PIL import Image
from scipy import ndimage
import imageio_ffmpeg

SRC_W, SRC_H = 1920, 1080
CARD        = 600           # output is CARD x CARD
MIN_W       = 1080          # tightest crop: full height, her largest
MAX_W       = 1600          # widest we go; past this the hands may leave
MARGIN      = 60            # air beyond the furthest bright pixel
LUMA        = 60            # brighter than this is her, not blazer or backdrop
BODY_LUMA   = 12            # above the backdrop's noise: blazer and sleeves count here
MIN_PX      = 3             # bright pixels a column needs (at quarter res) to count
ATTACK_S    = 0.20          # how fast the frame may widen
RELEASE_S   = 1.80          # how slowly it tightens again
LOOKAHEAD_S = 0.30          # widen a little before the hand arrives
SMOOTH_S    = 0.16          # final low-pass, so the zoom is continuous

FF = imageio_ffmpeg.get_ffmpeg_exe()


def fps_of(src):
    info = subprocess.run([FF, "-i", src], capture_output=True, text=True).stderr
    m = re.search(r"(\d+(?:\.\d+)?) fps", info)
    return float(m.group(1)) if m else 25.0


def measure(src, fps, start, dur):
    """Per frame, the crop width that would hold every bright pixel."""
    S = 4
    cut = (["-ss", str(start)] if start else []) + (["-t", str(dur)] if dur else [])
    raw = subprocess.run(
        [FF, "-v", "error", *cut, "-i", src,
         "-vf", f"scale={SRC_W//S}:{SRC_H//S},format=gray",
         "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"],
        capture_output=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, SRC_H // S, SRC_W // S)
    widths = np.full(len(fr), MIN_W, float)
    c = SRC_W / 2
    for i, f in enumerate(fr):
        # Her silhouette is whatever is above backdrop level *and* connected to
        # the bottom edge, where her body leaves the frame. That keeps a hand
        # at the end of a black sleeve and drops anything floating free — the
        # render's fade-in leaves a grey block at the right for its first two
        # seconds, and it must not pull the frame wide over the title.
        labels, n = ndimage.label(f > BODY_LUMA)
        if n == 0:
            continue
        keep = np.unique(labels[-1, :]); keep = keep[keep > 0]
        if len(keep) == 0:
            continue
        her = np.isin(labels, keep) & (f > LUMA)
        row = her.sum(axis=0) >= MIN_PX
        cols = np.flatnonzero(row)
        if len(cols) == 0:
            continue
        left, right = cols[0] * S, (cols[-1] + 1) * S
        widths[i] = 2 * max(c - left, right - c) + 2 * MARGIN
    return np.clip(widths, MIN_W, MAX_W)


def smooth(raw, fps):
    n = len(raw)
    la = max(1, int(LOOKAHEAD_S * fps))
    ahead = np.array([raw[i:i + la].max() for i in range(n)])
    att = (MAX_W - MIN_W) / (ATTACK_S * fps)
    rel = (MAX_W - MIN_W) / (RELEASE_S * fps)
    env = np.empty(n)
    cur = ahead[0]
    for i in range(n):
        target = ahead[i]
        if target > cur:  cur = min(target, cur + att)
        else:             cur = max(target, cur - rel)
        env[i] = cur
    k = max(1, int(SMOOTH_S * fps))
    ker = np.ones(k) / k
    return np.convolve(np.pad(env, (k // 2, k - 1 - k // 2), mode="edge"), ker, mode="valid")


def clean_opening(frame, x_from):
    """Black out everything right of x_from except her.

    Some renders open with a grey block at the right for their first seconds,
    a leftover of the engine's fade-in. It sits where a wide welcome gesture
    puts her hand, so it cannot simply be cut by column: keep the large bright
    blobs (skin), drop the block's dark body and its thin bright edges.
    """
    region = frame[:, x_from:]
    luma = region.mean(axis=2)
    keep = ndimage.binary_opening(luma > 70, structure=np.ones((9, 9)))
    keep = ndimage.binary_dilation(keep, iterations=3)
    region[~keep] = 0


def render(src, out, widths, fps, start, dur, clean=None):
    cut = (["-ss", str(start)] if start else []) + (["-t", str(dur)] if dur else [])
    dec = subprocess.Popen(
        [FF, "-v", "error", *cut, "-i", src, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE, bufsize=SRC_W * SRC_H * 3 * 4)
    enc = subprocess.Popen(
        [FF, "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{CARD}x{CARD}", "-r", str(fps), "-i", "pipe:0",
         *cut, "-i", src,
         "-map", "0:v", "-map", "1:a?", "-shortest",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "160k", out],
        stdin=subprocess.PIPE)
    frame_bytes = SRC_W * SRC_H * 3
    i = 0
    while True:
        buf = dec.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        w = int(round(widths[min(i, len(widths) - 1)]))
        w += w & 1                                   # even, for the encoder's sake
        x0 = SRC_W // 2 - w // 2
        y0 = SRC_H - w                               # bottom-anchored; negative = headroom
        frame = np.frombuffer(buf, np.uint8).reshape(SRC_H, SRC_W, 3)
        if clean and (i / fps) < clean["until"]:
            frame = frame.copy()
            clean_opening(frame, clean["x"])
        canvas = np.zeros((w, w, 3), np.uint8)
        top = max(0, y0)
        canvas[top - y0:, :] = frame[top:, x0:x0 + w]
        card = Image.fromarray(canvas).resize((CARD, CARD), Image.Resampling.HAMMING)
        enc.stdin.write(card.tobytes())
        i += 1
    dec.stdout.close(); enc.stdin.close()
    dec.wait(); enc.wait()
    if enc.returncode:
        sys.exit(f"encode failed for {out}")
    return i


def track(src, out, start=None, dur=None, report=None, clean=None):
    fps = fps_of(src)
    raw = measure(src, fps, start, dur)
    widths = smooth(raw, fps)
    n = render(src, out, widths, fps, start, dur, clean=clean)
    stats = {"frames": n, "fps": fps,
             "tight_share": float((widths <= MIN_W + 2).mean()),
             "max_width": float(widths.max()),
             "clipped_share": float((raw >= MAX_W).mean())}
    if report:
        json.dump({"widths": [round(float(w)) for w in widths], **stats}, open(report, "w"))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--start", type=float); ap.add_argument("--dur", type=float)
    ap.add_argument("--report", help="write per-frame widths and stats as JSON")
    ap.add_argument("--clean", metavar="SECONDS:X",
                    help="for the first SECONDS, black out right of X except her (see clean_opening)")
    a = ap.parse_args()
    clean = None
    if a.clean:
        secs, x = a.clean.split(":"); clean = {"until": float(secs), "x": int(x)}
    s = track(a.src, a.out, a.start, a.dur, a.report, clean=clean)
    print(f"{s['frames']} frames @ {s['fps']:g} fps; tight {s['tight_share']*100:.0f}% of the time, "
          f"widest {s['max_width']:.0f}px, hands beyond reach {s['clipped_share']*100:.1f}%")
