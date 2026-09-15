#!/usr/bin/env python3
"""Put the presenter straight onto the slide — no card.

The render is her on a backdrop that measures pure black, so she can be
keyed: everything above backdrop level that is connected to the bottom edge
(where her body leaves the frame) is her; the edge is eroded a pixel to
drop the dark fringe, then feathered. She is scaled and placed so she
stands in the slide's reserved right column, large, with nothing behind
her — a hand that goes wide simply passes over the slide's margin, as a
presenter's would.

    python3 tools/cutout.py media/part-2.webm slide.png out.mp4 --start 55 --dur 25
"""
import argparse, os, re, subprocess, sys
import numpy as np
from PIL import Image
from scipy import ndimage
import imageio_ffmpeg

SRC_W, SRC_H = 1920, 1080
OUT_W, OUT_H = 1920, 1080
SCALE     = 0.76            # her height on the slide: 0.76 * 1080 = 821 px
CENTER_X  = 1590            # where her body centre stands: in the right column, clear of the slide's photos
KEY_LUMA  = 3               # above this is not backdrop (backdrop measures exactly 0)
ERODE     = 2               # at half resolution: ~4 source px of dark fringe dropped
FEATHER   = 0.9             # sigma at half resolution of the edge softening
FF = imageio_ffmpeg.get_ffmpeg_exe()


def fps_of(src):
    info = subprocess.run([FF, "-i", src], capture_output=True, text=True).stderr
    m = re.search(r"(\d+(?:\.\d+)?) fps", info)
    return float(m.group(1)) if m else 25.0


def matte(frame):
    """Alpha in 0..1 at half resolution (SRC_H//2 x SRC_W//2).

    Half resolution is plenty: she is scaled down for the slide anyway, and
    the soft edge hides the difference. It also makes this four times faster.
    """
    small = frame[::2, ::2].max(axis=2)
    fg = small > KEY_LUMA
    # close first: the blazer's darkest edge pixels fall under the key and
    # would leave a ragged outline; a 7x7 close (14 source px) smooths it
    fg = ndimage.binary_closing(fg, structure=np.ones((7, 7)))
    # only what is connected to the bottom edge is her; stray specks are not
    labels, n = ndimage.label(fg)
    if n:
        keep = np.unique(labels[-1, :]); keep = keep[keep > 0]
        if len(keep):
            fg = np.isin(labels, keep)
    if ERODE:
        fg = ndimage.binary_erosion(fg, iterations=ERODE)          # the black edge blend
    return np.clip(ndimage.gaussian_filter(fg.astype(np.float32), FEATHER), 0, 1)


def place(frame, alpha, scale, center_x):
    """Scale her and return (rgb, alpha) canvases the size of the output."""
    w, h = int(round(SRC_W * scale)), int(round(SRC_H * scale))
    rgb = np.asarray(Image.fromarray(frame).resize((w, h), Image.Resampling.HAMMING), np.float32)
    al  = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR), np.float32) / 255
    ox, oy = int(round(center_x - w / 2)), OUT_H - h
    crgb = np.zeros((OUT_H, OUT_W, 3), np.float32); cal = np.zeros((OUT_H, OUT_W), np.float32)
    x0, x1 = max(ox, 0), min(ox + w, OUT_W)
    crgb[oy:, x0:x1] = rgb[:, x0 - ox:x1 - ox]
    cal[oy:, x0:x1]  = al[:, x0 - ox:x1 - ox]
    return crgb, cal


def composite(avatar, slide_src, out, start=None, dur=None, scale=SCALE, center_x=CENTER_X, clean=None):
    """slide_src is a still PNG or a video of the same length; avatar is the render."""
    fps = fps_of(avatar)
    cut = (["-ss", str(start)] if start else []) + (["-t", str(dur)] if dur else [])
    dec = subprocess.Popen([FF, "-v", "error", *cut, "-i", avatar, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                           stdout=subprocess.PIPE, bufsize=SRC_W * SRC_H * 3 * 4)
    sdec = None
    if slide_src.lower().endswith(".png"):
        slide = np.asarray(Image.open(slide_src).convert("RGB").resize((OUT_W, OUT_H)), np.float32)
    else:
        sdec = subprocess.Popen([FF, "-v", "error", "-i", slide_src, "-f", "rawvideo", "-pix_fmt", "rgb24",
                                 "-s", f"{OUT_W}x{OUT_H}", "-r", str(fps), "pipe:1"],
                                stdout=subprocess.PIPE, bufsize=OUT_W * OUT_H * 3 * 4)
    enc = subprocess.Popen([FF, "-y", "-v", "error",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OUT_W}x{OUT_H}", "-r", str(fps), "-i", "pipe:0",
                            *cut, "-i", avatar, "-map", "0:v", "-map", "1:a?", "-shortest",
                            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    if clean:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from track_presenter import clean_opening
    nbytes, sbytes, i = SRC_W * SRC_H * 3, OUT_W * OUT_H * 3, 0
    while True:
        buf = dec.stdout.read(nbytes)
        if len(buf) < nbytes: break
        frame = np.frombuffer(buf, np.uint8).reshape(SRC_H, SRC_W, 3)
        if clean and (i / fps) < clean["until"]:
            frame = frame.copy(); clean_opening(frame, clean["x"])
        if sdec:
            sb = sdec.stdout.read(sbytes)
            if len(sb) < sbytes: break
            slide = np.frombuffer(sb, np.uint8).reshape(OUT_H, OUT_W, 3).astype(np.float32)
        rgb, al = place(frame, matte(frame), scale, center_x)
        al3 = al[..., None]
        enc.stdin.write((slide * (1 - al3) + rgb * al3).astype(np.uint8).tobytes()); i += 1
    dec.stdout.close(); enc.stdin.close(); dec.wait(); enc.wait()
    if sdec: sdec.stdout.close(); sdec.wait()
    if enc.returncode: sys.exit(f"encode failed for {out}")
    return i


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("avatar"); ap.add_argument("slide"); ap.add_argument("out")
    ap.add_argument("--start", type=float); ap.add_argument("--dur", type=float)
    ap.add_argument("--scale", type=float, default=SCALE); ap.add_argument("--center", type=int, default=CENTER_X)
    a = ap.parse_args()
    n = composite(a.avatar, a.slide, a.out, a.start, a.dur, a.scale, a.center)
    print(f"{n} frames -> {a.out}")
