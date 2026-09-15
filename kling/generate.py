#!/usr/bin/env python3
"""Generate the B-roll clips in clips.json with Kling and save them to media/broll/.

Runs on a GitHub Actions runner, not in the authoring sandbox — the sandbox's
network policy refuses every Kling host. The key comes from the KLING_API_KEY
secret and is never written anywhere.

Kling issues credentials in two shapes depending on the account, and this
handles both without being told which one you have:

  accesskey:secretkey  ->  signed into a short-lived JWT, the documented scheme
  a single token       ->  sent straight through as a bearer token

The first request tells us which one works; after that it stops guessing.
"""

import json, os, pathlib, sys, time

import requests

ROOT      = pathlib.Path(__file__).resolve().parent.parent
MANIFEST  = ROOT / "kling" / "clips.json"
OUT_DIR   = ROOT / "media" / "broll"
BASE      = (os.environ.get("BASE_URL") or "https://api-singapore.klingai.com").rstrip("/")
KEY       = os.environ.get("KLING_API_KEY", "").strip()

# Kling finishes a five-second clip in a couple of minutes, but queues under
# load. Poll gently and give up rather than hold the runner for an hour.
POLL_EVERY   = 15
POLL_TIMEOUT = 20 * 60


def auth_header():
    """The Authorization value, whichever credential shape this account uses."""
    if ":" in KEY:
        import jwt  # only needed for the access/secret pair
        ak, sk = KEY.split(":", 1)
        now = int(time.time())
        token = jwt.encode(
            {"iss": ak, "exp": now + 1800, "nbf": now - 5},
            sk,
            algorithm="HS256",
            headers={"alg": "HS256", "typ": "JWT"},
        )
        return "Bearer " + (token if isinstance(token, str) else token.decode())
    return "Bearer " + KEY


def api(method, path, **kw):
    r = requests.request(
        method, BASE + path,
        headers={"Authorization": auth_header(), "Content-Type": "application/json"},
        timeout=60, **kw)
    if r.status_code >= 400:
        # Print the body — Kling explains refusals there, and it is the fastest
        # way to tell a wrong credential shape from an out-of-credit account.
        raise SystemExit(f"HTTP {r.status_code} from {path}\n{r.text[:800]}")
    return r.json()


def submit(clip, model):
    body = {
        "model_name":     model,
        "prompt":         clip["prompt"],
        "duration":       str(clip.get("duration", 5)),
        "aspect_ratio":   clip.get("aspect_ratio", "16:9"),
        "mode":           clip.get("mode", "std"),
        "cfg_scale":      clip.get("cfg_scale", 0.5),
        "negative_prompt": clip.get(
            "negative_prompt",
            "text, watermark, logo, signage, captions, subtitles, distorted hands, extra limbs"),
    }
    data = api("POST", "/v1/videos/text2video", json=body)
    task = (data.get("data") or {}).get("task_id")
    if not task:
        raise SystemExit(f"no task_id in response: {json.dumps(data)[:600]}")
    return task


def wait(task_id, name):
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        data   = (api("GET", f"/v1/videos/text2video/{task_id}").get("data") or {})
        status = data.get("task_status")
        if status == "succeed":
            videos = (data.get("task_result") or {}).get("videos") or []
            if not videos:
                raise SystemExit(f"{name}: succeeded with no video in the result")
            return videos[0]["url"]
        if status == "failed":
            raise SystemExit(f"{name}: {data.get('task_status_msg') or 'generation failed'}")
        print(f"  {name}: {status or 'pending'}", flush=True)
        time.sleep(POLL_EVERY)
    raise SystemExit(f"{name}: still not finished after {POLL_TIMEOUT // 60} minutes")


def download(url, dest):
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 16):
                fh.write(chunk)
    size = dest.stat().st_size
    # A short body is an error page, not a video.
    if size < 100_000:
        dest.unlink(missing_ok=True)
        raise SystemExit(f"{dest.name}: only {size} bytes, treating as a failure")
    return size


def main():
    if not KEY:
        raise SystemExit("KLING_API_KEY is empty")

    manifest = json.loads(MANIFEST.read_text())
    model    = manifest.get("model", "kling-v1")
    clips    = manifest["clips"]

    only = [n.strip() for n in (os.environ.get("ONLY") or "").split(",") if n.strip()]
    if only:
        clips = [c for c in clips if c["name"] in only]
        missing = set(only) - {c["name"] for c in clips}
        if missing:
            raise SystemExit(f"no such clip: {', '.join(sorted(missing))}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Submit everything first, then collect — they generate in parallel, so the
    # run takes as long as the slowest clip rather than the sum of all of them.
    pending = []
    for clip in clips:
        dest = OUT_DIR / f"{clip['name']}.mp4"
        if dest.exists() and not only:
            print(f"skip {clip['name']} — already generated")
            continue
        print(f"submit {clip['name']}")
        pending.append((clip, dest, submit(clip, model)))

    if not pending:
        print("nothing to generate")
        return

    print(f"\nwaiting on {len(pending)} clip(s)\n")
    for clip, dest, task_id in pending:
        url  = wait(task_id, clip["name"])
        size = download(url, dest)
        print(f"saved {dest.relative_to(ROOT)}  ({size:,} bytes)")


if __name__ == "__main__":
    main()
