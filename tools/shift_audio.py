#!/usr/bin/env python3
"""
shift_audio.py — move the narration earlier or later against the picture.

Lip sync is the one thing here that is cheap to change late: the audio is
independent of the expensive keying, so this rewrites only the sound and copies
the video stream through untouched. A twelve-minute course re-syncs in about a
minute, where re-compositing it would take hours.

Pulling the audio earlier costs nothing: every render opens with about a third
of a second of silence before her first word, so there is room to take from.
The same amount of silence is added at the end, which keeps the duration — and
so every knowledge-check gate — exactly where it was.

USAGE
    python3 tools/shift_audio.py in.mp4 out.mp4 --ms -80      # voice 80 ms earlier
    python3 tools/shift_audio.py in.mp4 out.mp4 --ms 40       # voice 40 ms later
"""

import argparse, os, shutil, subprocess, sys


def ffmpeg_bin():
    return shutil.which("ffmpeg") or __import__("imageio_ffmpeg").get_ffmpeg_exe()


def shift(ff, src, dst, ms):
    s = abs(ms) / 1000.0
    if ms < 0:      # earlier: drop from the head, make it up at the tail
        af = f"atrim=start={s:.4f},asetpts=PTS-STARTPTS,apad=pad_dur={s:.4f}"
    elif ms > 0:    # later: silence at the head, and let the tail run off the end
        af = f"adelay={int(round(s*1000))}:all=1"
    else:
        af = "anull"
    subprocess.run(
        [ff, "-y", "-loglevel", "error", "-i", src,
         "-af", af, "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
         "-movflags", "+faststart", "-shortest", dst],
        check=True)
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--ms", type=float, required=True,
                    help="negative pulls the voice earlier, positive pushes it later")
    args = ap.parse_args()
    ff = ffmpeg_bin()
    shift(ff, args.src, args.dst, args.ms)
    # the gate metadata travels with the file and is unchanged by a shift
    meta = os.path.splitext(args.src)[0] + ".json"
    if os.path.exists(meta):
        shutil.copy(meta, os.path.splitext(args.dst)[0] + ".json")
    print(f"{args.dst}  ({os.path.getsize(args.dst)/1e6:.1f} MB, voice {args.ms:+.0f} ms)")


if __name__ == "__main__":
    main()
