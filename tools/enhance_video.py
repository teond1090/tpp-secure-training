#!/usr/bin/env python3
"""
enhance_video.py — composite a rendered narration segment with its slides into
a finished, more dynamic training video.

The HeyGen render gives us a presenter on a flat background and nothing else.
This turns that into something that holds attention for twelve minutes:

  * slides are burned in underneath, changing on cue with the narration
  * each slide gets a slow Ken Burns push, so a static PNG is never truly still
  * slides cross-fade rather than cutting, which reads as produced rather than
    assembled
  * the presenter is cropped out of her wide empty frame, rounded, shadowed and
    docked in the reserved right-hand column
  * a light grade lifts the presenter, who renders slightly flat against the
    bright slides

The output is one self-contained MP4 per section, so the course player does not
have to composite anything at runtime and the file can also be emailed or
uploaded anywhere.

USAGE
    python3 tools/enhance_video.py --manifest tools/enhance.json --out build/
    python3 tools/enhance_video.py --manifest tools/enhance.json --only part-1
    python3 tools/enhance_video.py --manifest tools/enhance.json --preview 12
        (--preview N renders only the first N seconds — use it to check framing
         before committing to a full encode)

Requires ffmpeg. If it is not on PATH, `pip install imageio-ffmpeg` provides a
static build and this script will find it automatically.
"""

import argparse, json, os, shutil, subprocess, sys, tempfile

W, H = 1920, 1080

# The presenter card, in output pixels. Matches the reserved column in the
# slide layouts, so she never covers content.
# Square. The high-energy take opens her arms well past a portrait crop —
# a 560x653 card cut a hand off in one frame in five. Square shows both
# hands at almost the same size; anything wider shrinks her noticeably.
CARD_W, CARD_H = 600, 600
CARD_X, CARD_Y = W - CARD_W - 40, H - CARD_H - 40
CARD_MARGIN = 40
CARD_RADIUS = 28

# How much of the source frame to keep. The avatar sits centred in a 16:9 frame
# with wide empty margins; this crops to her.
# Wide enough to keep her hands in frame when she gestures — a tighter crop
# clips them at the card edge.
SRC_CROP_W, SRC_CROP_H = 1080, 1080
SRC_CROP_X, SRC_CROP_Y = (1920 - SRC_CROP_W) // 2, 0


def set_card(spec):
    """Re-derive the card and source crop from a WxH card size.

    The crop always takes the full 1080 source height at the card's aspect,
    centred — so a wider card shows more of her gesture range, at the cost of
    rendering her smaller. --card 600x600 gives a square card, for instance.
    """
    global CARD_W, CARD_H, CARD_X, CARD_Y, SRC_CROP_W, SRC_CROP_H, SRC_CROP_X, SRC_CROP_Y
    cw, ch = (int(v) for v in spec.lower().split("x"))
    CARD_W, CARD_H = cw, ch
    CARD_X, CARD_Y = W - cw - CARD_MARGIN, H - ch - CARD_MARGIN
    SRC_CROP_H = 1080
    SRC_CROP_W = min(1920, int(round(1080 * cw / ch)))
    SRC_CROP_X, SRC_CROP_Y = (1920 - SRC_CROP_W) // 2, 0

KENBURNS_ZOOM = 1.06      # how far each slide pushes in over its time on screen
XFADE = 0.6               # seconds of cross-fade between slides


def ffmpeg_bin():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        sys.exit("ffmpeg not found. Install it, or: pip install imageio-ffmpeg")


def probe_duration(ff, path):
    out = subprocess.run([ff, "-i", path], capture_output=True, text=True).stderr
    for line in out.splitlines():
        if "Duration:" in line:
            hms = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = hms.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    sys.exit(f"could not read duration of {path}")


def build_slide_track(ff, slides, total, slide_dir, tmp):
    """Render the slide sequence to its own video: Ken Burns push, cross-fades.

    Done as a separate pass because chaining twenty zoompans and nineteen
    xfades into the main graph makes a filter string long enough to trip
    ffmpeg's argument limits on some platforms.
    """
    # Convert the fractional cues into concrete spans.
    spans = []
    for i, s in enumerate(slides):
        start = s["at"] * total
        end = slides[i + 1]["at"] * total if i + 1 < len(slides) else total
        spans.append((s["img"], start, max(end - start, 1.0)))

    parts = []
    for idx, (img, _start, dur) in enumerate(spans):
        src = os.path.join(slide_dir, img)
        if not os.path.exists(src):
            sys.exit(f"missing slide: {src}")
        out = os.path.join(tmp, f"slide_{idx:03d}.mp4")
        frames = max(int(dur * 30), 2)
        # zoompan needs an oversized input or the push shows edge artefacts.
        vf = (
            f"scale={W*2}:{H*2},"
            f"zoompan=z='min(zoom+{(KENBURNS_ZOOM-1)/frames:.8f},{KENBURNS_ZOOM})'"
            f":d={frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={W}x{H}:fps=30,"
            f"setsar=1"
        )
        subprocess.run(
            [ff, "-y", "-loglevel", "error", "-loop", "1", "-i", src,
             "-t", f"{dur:.3f}", "-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "18", "-pix_fmt", "yuv420p", out],
            check=True,
        )
        parts.append((out, dur))

    if len(parts) == 1:
        return parts[0][0]

    # Cross-fade the clips together, one pair at a time.
    cur, cur_dur = parts[0]
    for i, (nxt, nxt_dur) in enumerate(parts[1:], start=1):
        out = os.path.join(tmp, f"xf_{i:03d}.mp4")
        offset = max(cur_dur - XFADE, 0)
        subprocess.run(
            [ff, "-y", "-loglevel", "error", "-i", cur, "-i", nxt,
             "-filter_complex",
             f"[0:v][1:v]xfade=transition=fade:duration={XFADE}:offset={offset:.3f},format=yuv420p[v]",
             "-map", "[v]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", out],
            check=True,
        )
        cur, cur_dur = out, offset + nxt_dur
    return cur


def compose(ff, seg, slide_track, out_path, preview=None, at=None):
    """Overlay the cropped, graded, rounded presenter card onto the slide track."""
    dur_arg = ["-t", str(preview)] if preview else []
    # --at T: a single PNG frame, with the slide passed in as a still image
    # rather than a rendered track. Cheap enough to try several card sizes.
    if at is not None:
        # Pull the frame out first and composite two stills. Feeding a looped
        # PNG and a seeked video straight into overlay is unreliable: the PNG's
        # first frame can reach the overlay before the video's does, and the
        # slide passes through with no card on it.
        frame = out_path + ".frame.png"
        subprocess.run([ff, "-y", "-loglevel", "error", "-ss", str(at), "-i", seg,
                        "-frames:v", "1", "-update", "1", frame], check=True)
        in0 = ["-i", slide_track]
        in1 = ["-i", frame]
        enc = ["-frames:v", "1", "-update", "1"]
    else:
        in0 = ["-i", slide_track]; in1 = ["-i", seg]
        enc = ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart"]

    # Rounded corners via geq on the alpha plane: inside the radius keep the
    # pixel, outside drop it. Cheaper and more portable than an RGBA PNG mask.
    r = CARD_RADIUS
    inside = (
        f"if(gt(abs(X-(W/2)),W/2-{r})*gt(abs(Y-(H/2)),H/2-{r}),"
        f"if(lte(pow(abs(X-(W/2))-(W/2-{r}),2)+pow(abs(Y-(H/2))-(H/2-{r}),2),{r*r}),255,0),255)"
    )
    filt = (
        f"[1:v]crop={SRC_CROP_W}:{SRC_CROP_H}:{SRC_CROP_X}:{SRC_CROP_Y},"
        f"scale={CARD_W}:{CARD_H},"
        f"eq=brightness=0.04:contrast=1.07:saturation=1.06,"      # lift a flat render
        f"format=rgba,geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='{inside}'[card];"
        f"[0:v][card]overlay={CARD_X}:{CARD_Y}:format=auto[v]"
    )
    subprocess.run(
        [ff, "-y", "-loglevel", "error",
         *in0, *in1, *dur_arg,
         "-filter_complex", filt,
         "-map", "[v]", *([] if at is not None else ["-map", "1:a?"]),
         *enc,
         out_path],
        check=True,
    )
    if at is not None:
        os.remove(out_path + ".frame.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="JSON describing segments and slide cues")
    ap.add_argument("--out", default="build", help="output directory")
    ap.add_argument("--only", help="build just this segment name")
    ap.add_argument("--preview", type=float, help="render only the first N seconds")
    ap.add_argument("--card", help="card size as WxH (default 600x600)")
    ap.add_argument("--at", type=float, help="write one PNG still at this second instead of a video")
    args = ap.parse_args()
    if args.card:
        set_card(args.card)

    ff = ffmpeg_bin()
    man = json.load(open(args.manifest))
    slide_dir = man["slide_dir"]
    os.makedirs(args.out, exist_ok=True)

    for seg in man["segments"]:
        name = seg["name"]
        if args.only and args.only != name:
            continue
        src = seg["video"]
        if not os.path.exists(src):
            print(f"  skip {name}: {src} not found")
            continue

        total = probe_duration(ff, src)
        print(f"{name}: {total:.1f}s, {len(seg['slides'])} slides")
        if args.at is not None:
            # the slide showing at that moment: last cue at or before it
            cue = max((c for c in seg["slides"] if c["at"] * total <= args.at),
                      key=lambda c: c["at"], default=seg["slides"][0])
            still = os.path.join(slide_dir, cue["img"])
            tag = f"-{args.card}" if args.card else ""
            out_path = os.path.join(args.out, f"{name}{tag}-at{int(args.at)}.png")
            compose(ff, src, still, out_path, at=args.at)
            print(f"  -> {out_path}")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            track = build_slide_track(ff, seg["slides"], total, slide_dir, tmp)
            out_path = os.path.join(args.out, f"{name}.mp4")
            compose(ff, src, track, out_path, args.preview)
        size = os.path.getsize(out_path) / 1e6
        print(f"  -> {out_path}  ({size:.1f} MB)")

    print("\nDone.")


if __name__ == "__main__":
    main()
