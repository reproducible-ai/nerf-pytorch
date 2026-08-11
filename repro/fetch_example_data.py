#!/usr/bin/env python
"""Fetch the two example scenes this repository's quick-start uses.

This is upstream's own `download_example_data.sh` expressed as a traced,
content-addressed Python step. The URL is byte-for-byte the one in that script:

    http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip

The archive holds exactly two scenes -- `nerf_llff_data/fern` (the LLFF
forward-facing example) and `nerf_synthetic/lego` (the Blender synthetic
example) -- which are the two scenes README.md tells a new user to train on.

Why a script instead of `bash download_example_data.sh`:

* **The archive is pinned by sha256.** The shell script pipes `wget` into
  `unzip` and accepts whatever the server returns. A dataset that is only
  identified by a URL is not a reproducible input: the file at that URL can
  change, and a rebuild would silently train on different data. `--sha256` makes
  the dataset a *content-addressed* node in the lineage rather than a name.
* **The output directory is an explicit argument.** The shell script `cd`s into
  a relative `data/`, which puts the destination outside the recorded command.
* **A shell step records no Python environment.** Every other step in this
  pipeline records the interpreter's loaded distributions; a `bash` step would
  contribute an empty package set to the record for no reason.
* `download_example_data.sh` additionally fetches `tiny_nerf_data.npz` into the
  repository root. Nothing in this repository reads that file -- it belongs to
  the authors' separate "tiny NeRF" notebook -- so it is not fetched here, and
  the repository root stays clean for the next traced step.

Nothing under this repository's model or training code is touched.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_URL = (
    "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip"
)
# sha256 of the archive as served on 2026-08-11. Recorded here so a rebuild
# proves it trained on the same bytes rather than on the same URL.
DEFAULT_SHA256 = "ce4e94e031c099a19ef04cfb6c71f1e47225d97d365be610b476e379a386c25f"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--sha256", default=DEFAULT_SHA256)
    ap.add_argument(
        "--out-dir",
        default="data",
        help="destination for the extracted scenes (upstream's configs expect ./data)",
    )
    cfg = ap.parse_args()

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / "nerf_example_data.zip"

    t0 = time.time()
    if archive.exists() and sha256_file(archive) == cfg.sha256:
        print(f"archive already present and verified: {archive}")
    else:
        print(f"downloading {cfg.url}")
        with urllib.request.urlopen(cfg.url, timeout=600) as resp:  # noqa: S310
            with archive.open("wb") as fh:
                while True:
                    block = resp.read(1 << 20)
                    if not block:
                        break
                    fh.write(block)
        print(f"downloaded {archive.stat().st_size} bytes in {time.time() - t0:.1f}s")

    got = sha256_file(archive)
    print(f"expected sha256: {cfg.sha256}")
    print(f"actual   sha256: {got}")
    if got != cfg.sha256:
        print("FAIL: the archive served does not match the recorded digest")
        return 1

    t1 = time.time()
    with zipfile.ZipFile(archive) as zf:
        members = zf.namelist()
        zf.extractall(out_dir)
    print(f"extracted {len(members)} entries into {out_dir} in {time.time() - t1:.1f}s")

    total = 0
    for root, _dirs, files in os.walk(out_dir):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    print(f"{out_dir} now holds {total} bytes")

    # The two scenes upstream's configs point at must both be on disk, or the
    # train step would fail several minutes later with a bare FileNotFoundError.
    for expected in ("nerf_synthetic/lego/transforms_train.json", "nerf_llff_data/fern/poses_bounds.npy"):
        path = out_dir / expected
        if not path.exists():
            print(f"FAIL: expected {path} in the extracted archive")
            return 1
        print(f"ok: {path}")

    print(f"fetch_example_data: PASS ({time.time() - t0:.1f}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
