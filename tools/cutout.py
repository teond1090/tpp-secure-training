#!/usr/bin/env python3
"""Put the presenter straight onto the slide — no card.

HeyGen ships these renders as VP9 WebM with a real alpha channel — the
container says alpha_mode 1 — so the matte is exact and free. It has to be
asked for: ffmpeg's native VP9 decoder silently discards the alpha plane and
hands back a fully opaque frame, which is what made this file look like a
plain black-backdrop render. Decoding through libvpx-vp9 returns it.

That matters because she wears a black blazer against a black backdrop, and
parts of her measure exactly 0 — identical to the backdrop. No threshold can
separate those, so the keyer below eroded her edges, crawled from frame to
frame and let the slide flash through her clothes. The shipped matte has none
of that: fully opaque inside her, exactly zero outside, cleanly anti-aliased
between. The keyer is kept only for a source that genuinely has no alpha.

She is scaled and placed so she stands in the slide's reserved right column,
large, with nothing behind her — a hand that goes wide simply passes over the
slide's margin, as a presenter's would.

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
KEY_LUMA  = 4               # above the compression noise in the fringe, below her darkest cloth
LUMA_BLUR = 1.2             # px; steadies the contour so the sleeve edge stops crawling
MIN_PIECE = 400             # px; a piece this big is part of her even if it looks detached
SHRINK    = 5.0             # px to pull the contour in, past the render's dark fringe
SOFT      = 1.2             # px the alpha ramp takes to cross from 0 to 1
# Off. It was meant to ground her, but at 0.14 over a white slide it read as a
# dirty grey outline hugging her contour: measured on a plain white background
# it pulled the slide from 250 down to 209 right at her edge and still had not
# recovered 20px out. With an exact matte she does not need the help.
SHADOW    = 0.0             # how dark her contact shadow sits on the slide, 0 for none
SHADOW_BLUR = 26            # px of softness on that shadow
SHADOW_DX, SHADOW_DY = 14, 10   # px the shadow is offset, as if the key light is high and left
GHOST_CORE  = 5             # px; a lump this thin at its waist is not a limb of hers
GHOST_REACH = 10            # px her contour may sit outside its own core
GHOST_FRAC  = 0.30          # a core under this share of her core is not part of her
GHOST_KEEP  = 40            # true colour thin fringe may not be removed above: hair and skin
GHOST_STEP  = 2             # the shape test runs on every other pixel; 6x cheaper, same verdict
GHOST_MIN   = 25            # px; below this a piece is a speck, not worth colour-testing
GHOST_SOLID = 0.10          # alpha a piece must reach to be removed at all
GHOST_SKIRT = 14            # px of its own soft edge that come away with a piece
FF = imageio_ffmpeg.get_ffmpeg_exe()


def has_alpha(src):
    """True when the WebM carries its own matte (alpha_mode 1 in the container)."""
    info = subprocess.run([FF, "-i", src], capture_output=True, text=True).stderr
    return "alpha_mode" in info and re.search(r"alpha_mode\s*:\s*1", info) is not None


def fps_of(src):
    info = subprocess.run([FF, "-i", src], capture_output=True, text=True).stderr
    m = re.search(r"(\d+(?:\.\d+)?) fps", info)
    return float(m.group(1)) if m else 25.0


def matte(frame):
    """Alpha in 0..1 at full resolution, anti-aliased.

    Two things made her look grainy on a white slide.

    The first was the matte's shape. Thresholding to a binary mask and then
    blurring leaves the blur following whole pixels, so every edge carries a
    staircase — most visible along her hair. A signed distance transform gives
    a ramp that crosses 0.5 on the true contour instead, so the edge stays
    smooth however far the frame is scaled.

    The second was the render's own dark fringe. She is rendered over black,
    and the outermost five or six pixels of her silhouette are part her and
    part backdrop: measured across her hair, luma climbs 1, 2, 6, 13, 22, 26
    before reaching her real brightness. Keeping those pixels puts a grey rim
    around her on a white slide, and keying cannot tell them from genuinely
    dark hair. The contour is therefore pulled in past the fringe.

    Holes are filled before the distance transform so the black inside the
    blazer does not read as background.
    """
    lum = frame.max(axis=2).astype(np.float32)
    if LUMA_BLUR:
        # The fringe is only a few levels above black, so per-pixel compression
        # noise moved the contour around from frame to frame and the sleeve
        # edge crawled. Smoothing the luma first settles it.
        lum = ndimage.gaussian_filter(lum, LUMA_BLUR)
    fg = lum > KEY_LUMA
    # Keep what reaches the bottom edge, and any other piece big enough to be
    # part of her. Keying higher up pinched a dark sleeve off from the body on
    # some frames, and a bottom-edge-only rule then threw that piece away for
    # exactly those frames — an arm blinking in and out. The backdrop is
    # exactly 0, so nothing above it is background and the test can be this
    # generous without letting the render's fade-in artefacts through.
    labels, n = ndimage.label(fg)
    if n:
        keep = set(np.unique(labels[-1, :])) - {0}
        sizes = ndimage.sum(fg, labels, range(1, n + 1))
        keep |= {i + 1 for i, sz in enumerate(sizes) if sz >= MIN_PIECE}
        if keep:
            fg = np.isin(labels, list(keep))
    fg = ndimage.binary_fill_holes(fg)

    # The distance transform is the expensive step and only matters near her,
    # so run it on her bounding box with a margin wider than the ramp. Outside
    # that box the alpha is 0 regardless, so the result is identical.
    ys, xs = np.nonzero(fg)
    if len(ys) == 0:
        return np.zeros(fg.shape, np.float32)
    pad = int(SHRINK + 4 * SOFT + 2)
    y0, y1 = max(0, ys.min() - pad), min(fg.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(fg.shape[1], xs.max() + pad + 1)
    sub = fg[y0:y1, x0:x1]
    d = ndimage.distance_transform_edt(sub) - ndimage.distance_transform_edt(~sub)
    a = np.zeros(fg.shape, np.float32)
    a[y0:y1, x0:x1] = np.clip((d - SHRINK) / (2 * SOFT) + 0.5, 0, 1)
    return a


def despeckle(rgb, a):
    """Drop the render's ghost limbs before anything else looks at the matte.

    HeyGen's render carries motion ghosts around a moving hand: a blurred lump
    that the matte then makes opaque. Measured on part-2b at 48-49s, lumps of
    up to 4400 px sit beside her hand at alpha 1.00 over a colour that averages
    10 of 255. On the black backdrop they are invisible — black on black — so
    they survive review of the source, and on a white slide each one paints a
    hard grey smudge that swells and vanishes over about six frames. That is
    what read as a shadow flickering under her arm.

    Neither hardening nor a gentler resample touches this: the matte is not
    soft there, it is confidently wrong. Colour cannot select it either, since
    a ghost measures 4 to 31 against a blazer at 21.

    Shape can. She is one solid body, so every part of her is reachable from
    her core — what survives eroding the matte by GHOST_CORE. A ghost has its
    own little core, either free-floating or hanging off her by a neck too thin
    to be a wrist. Over 40 frames of that passage this rejected 29 regions, all
    of them ghosts of true colour 4 to 31; nothing skin- or hair-coloured was
    touched. The colour test is kept only as a one-way guard, so that if the
    shape test ever misfires the piece is kept.
    """
    m = a > 0.5
    if not m.any():
        return rgb, a
    # Erode and dilate by distance rather than by repeated 3x3 passes: one pass
    # each, and a true disc, so a diagonal neck is measured the same as a
    # straight one. Both run on every other pixel, which is six times cheaper
    # and agrees with the full-resolution verdict; the coarse grid is paid back
    # by widening her afterwards, so the error can only ever keep more of her.
    # The mask is edge-padded first, because she runs off the bottom of the
    # frame and must not be eroded inward from that edge.
    s = GHOST_STEP
    pad = (GHOST_CORE + GHOST_REACH) // s + 2
    mp = np.pad(m[::s, ::s], pad, mode="edge")
    core = ndimage.distance_transform_edt(mp) > GHOST_CORE / s
    lab, n = ndimage.label(core)
    if n < 2:
        return rgb, a
    sizes = ndimage.sum(core, lab, index=range(1, n + 1))
    good = np.nonzero(sizes >= sizes.max() * GHOST_FRAC)[0] + 1
    if len(good) == n:
        return rgb, a
    body = ndimage.distance_transform_edt(~np.isin(lab, good)) <= GHOST_REACH / s
    body = ndimage.binary_dilation(body)[pad:-pad, pad:-pad]   # pay back the coarse grid
    body = np.repeat(np.repeat(body, s, 0), s, 1)[:m.shape[0], :m.shape[1]]
    # Only material that is actually opaque is a candidate. Her silhouette
    # carries a very faint halo — alpha around 0.03, two hundred-odd specks a
    # frame — and sweeping that away here but not on the frames with no ghost
    # to find would have made her outline flicker at exactly the rate of the
    # fault being fixed. Each rejected piece still takes its own soft edge with
    # it, so nothing is left tracing where it was.
    kill = (a > GHOST_SOLID) & ~body
    if not kill.any():
        return rgb, a
    # Thin fringe that reads as hair or skin is hers whatever its shape says.
    # The guard stops at fringe: a piece with a core of its own is a solid
    # object, and she carries none — the ghosts are not all dark, and a
    # light-grey rounded lump detaches from her hand in RV part 5 at frame
    # 4863, 3508 px of core at colour 191. Protecting anything pale would have
    # kept that. Hair wisps have no core, being thin, so they keep their guard.
    #
    # Only the pieces worth looking at are tested, and each inside its own
    # bounding box: a frame carries a few hundred single-pixel specks, and
    # touching the whole frame once per speck cost more than everything else
    # here together.
    solid = np.repeat(np.repeat((lab > 0)[pad:-pad, pad:-pad], s, 0), s, 1)[:m.shape[0], :m.shape[1]]
    lab, n = ndimage.label(kill)
    sizes = np.bincount(lab.ravel())
    boxes = ndimage.find_objects(lab)
    for j in range(1, n + 1):
        if sizes[j] < GHOST_MIN:
            continue
        sl = boxes[j - 1]
        piece = lab[sl] == j
        if (piece & solid[sl]).any():
            continue                            # has a core of its own: not fringe
        lit = piece & (a[sl] > 0.4)
        if lit.any() and np.median((rgb[sl].mean(2) / np.maximum(a[sl], 1e-3))[lit]) >= GHOST_KEEP:
            kill[sl] &= ~piece
    if not kill.any():
        return rgb, a
    ys, xs = np.nonzero(kill)                    # dilate only where there is something to grow
    y0, y1 = max(0, ys.min() - GHOST_SKIRT), min(kill.shape[0], ys.max() + GHOST_SKIRT + 1)
    x0, x1 = max(0, xs.min() - GHOST_SKIRT), min(kill.shape[1], xs.max() + GHOST_SKIRT + 1)
    win = (slice(y0, y1), slice(x0, x1))
    kill[win] = ndimage.binary_dilation(kill[win], iterations=GHOST_SKIRT)
    kill &= ~body & (a > 0)
    rgb = rgb.copy(); a = a.copy()
    rgb[kill] = 0
    a[kill] = 0.0
    return rgb, a


def harden(rgb, a):
    """Force her interior to full opacity, keeping the contour anti-aliased.

    HeyGen's matte is less certain on near-black cloth: measured over her lower
    body, 1242 pixels a frame sit between 0.93 and 0.99 alpha well inside her
    blazer. Each one lets a little of the white slide through, and because the
    values wobble frame to frame it reads as white pixels crawling in the hem.

    A pixel whose whole 3x3 neighbourhood is solid cannot be on the contour, so
    it is pushed to 1. The colour is scaled by the same factor, because the
    render is premultiplied and the two have to stay in step.
    """
    core = ndimage.grey_erosion(a, size=3) > 0.6
    if not core.any():
        return rgb, a
    gain = np.where(core, 1.0 / np.maximum(a, 1e-3), 1.0)
    np.minimum(gain, 1.5, out=gain)             # never invent more than a nudge
    rgb = np.clip(rgb.astype(np.float32) * gain[..., None], 0, 255)
    return rgb, np.where(core, 1.0, a)


def place(frame, alpha, scale, center_x):
    """Scale her and return (rgb, alpha) canvases the size of the output."""
    w, h = int(round(SRC_W * scale)), int(round(SRC_H * scale))
    # One kernel for both: premultiplied colour and its alpha have to be scaled
    # together or the relationship between them breaks at every edge. Hamming
    # also does not overshoot, which Lanczos does — an overshoot on a matte
    # shows up as a bright halo.
    rgb = np.asarray(Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8))
                     .resize((w, h), Image.Resampling.HAMMING), np.float32)
    al  = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8))
                     .resize((w, h), Image.Resampling.HAMMING), np.float32) / 255
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
    # libvpx-vp9 rather than the native decoder: the native one drops the alpha
    # plane without saying so, and hands back an opaque frame.
    alpha = has_alpha(avatar)
    chans = 4 if alpha else 3
    dec = subprocess.Popen(
        [FF, "-v", "error", *(["-c:v", "libvpx-vp9"] if alpha else []), *cut, "-i", avatar,
         "-f", "rawvideo", "-pix_fmt", "rgba" if alpha else "rgb24", "pipe:1"],
        stdout=subprocess.PIPE, bufsize=SRC_W * SRC_H * chans * 4)
    sdec = None
    if slide_src.lower().endswith(".png"):
        slide = np.asarray(Image.open(slide_src).convert("RGB").resize((OUT_W, OUT_H)), np.float32)
    else:
        sdec = subprocess.Popen([FF, "-v", "error", "-i", slide_src, "-f", "rawvideo", "-pix_fmt", "rgb24",
                                 "-s", f"{OUT_W}x{OUT_H}", "-r", str(fps), "pipe:1"],
                                stdout=subprocess.PIPE, bufsize=OUT_W * OUT_H * 3 * 4)
        slide = np.zeros((OUT_H, OUT_W, 3), np.float32)
    enc = subprocess.Popen([FF, "-y", "-v", "error",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OUT_W}x{OUT_H}", "-r", str(fps), "-i", "pipe:0",
                            *cut, "-i", avatar, "-map", "0:v", "-map", "1:a?", "-shortest",
                            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    if clean:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from track_presenter import clean_opening
    nbytes, sbytes, i = SRC_W * SRC_H * chans, OUT_W * OUT_H * 3, 0
    slide_done = False
    while True:
        buf = dec.stdout.read(nbytes)
        if len(buf) < nbytes: break
        arr = np.frombuffer(buf, np.uint8).reshape(SRC_H, SRC_W, chans)
        if alpha:
            frame, a = np.ascontiguousarray(arr[..., :3]), arr[..., 3].astype(np.float32) / 255
            frame, a = despeckle(frame, a)
            frame, a = harden(frame, a)
        else:
            frame, a = arr, matte(arr)
        if clean and (i / fps) < clean["until"]:
            frame = frame.copy(); clean_opening(frame, clean["x"])
        if sdec and not slide_done:
            sb = sdec.stdout.read(sbytes)
            if len(sb) < sbytes:
                # The slides ran out before she stopped talking. Hold the last
                # one rather than stopping: ending here truncates the narration
                # mid-sentence, which is exactly the fault this guards against.
                slide_done = True
            else:
                slide = np.frombuffer(sb, np.uint8).reshape(OUT_H, OUT_W, 3).astype(np.float32)
        rgb, al = place(frame, a, scale, center_x)
        al3 = al[..., None]
        base = slide
        if SHADOW:
            # A soft shadow cast onto the slide, so she stands on it rather
            # than floating above it. Offset down and right of her, as if the
            # key light were high and to the left.
            # shift, not roll: she reaches the bottom of the frame, and rolling
            # wrapped that edge around into a grey band across the top
            sh = ndimage.shift(al, (SHADOW_DY, SHADOW_DX), order=0, mode="constant", cval=0.0)
            sh = ndimage.gaussian_filter(sh, SHADOW_BLUR) * SHADOW
            base = slide * (1 - sh[..., None])
        # The render's alpha is premultiplied — the stored colour is already
        # scaled by it (fitted over her hair, rgb = 48.2 * alpha, against a flat
        # fit 33x worse). Multiplying by alpha again darkened every soft pixel,
        # which is the black line that was tracing her hair.
        px = base * (1 - al3) + (rgb if alpha else rgb * al3)
        enc.stdin.write(np.clip(px, 0, 255).astype(np.uint8).tobytes()); i += 1
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
