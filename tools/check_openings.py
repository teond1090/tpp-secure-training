#!/usr/bin/env python3
"""Does any render open with the engine's grey fade-in block at the right?

Prints, per part, the bright pixels in the block's region over the first
four seconds. A run of thousands that drops to zero by ~3.5 s is the block;
add a "clean" rule for that segment in enhance.json. Hands passing through
the region show as brief spikes later, not a steady run from t=0.
"""
import json, subprocess, sys, numpy as np, imageio_ffmpeg
FF = imageio_ffmpeg.get_ffmpeg_exe()
man = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "tools/enhance.json"))
for seg in man["segments"]:
    out = []
    for t in (0.2, 1, 2, 3, 3.5, 4):
        raw = subprocess.run([FF, "-v", "error", "-ss", str(t), "-i", seg["video"], "-frames:v", "1",
                              "-vf", "format=gray", "-f", "rawvideo", "pipe:1"], capture_output=True).stdout
        f = np.frombuffer(raw, np.uint8).reshape(1080, 1920)
        out.append(f"{t}s:{(f[500:1000, 1480:1900] > 60).sum()}")
    flag = "  <-- block?" if int(out[0].split(':')[1]) > 500 and int(out[1].split(':')[1]) > 500 else ""
    print(f"{seg['name']}: " + " ".join(out) + flag)
