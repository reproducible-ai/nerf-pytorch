#!/usr/bin/env python
"""Report the native shared libraries this environment needs but pip cannot record.

`requirements.txt` lists `opencv-python` and `imageio-ffmpeg`. Both are Python
distributions, so both appear in a pip freeze -- and a reader of that freeze
would reasonably conclude the environment is fully described by it. For one of
them that is true and for the other it is not, and the difference is invisible
from the freeze alone:

* **imageio-ffmpeg** ships a *statically linked* `ffmpeg` executable inside the
  wheel. Installing the wheel installs FFmpeg. The freeze is sufficient.
* **opencv-python** vendors most of its shared libraries into
  `opencv_python.libs/`, but `cv2`'s extension module still records unbundled
  `NEEDED` entries for the system OpenGL and glib stacks (`libGL.so.1`,
  `libGLdispatch.so.0`, `libGLX.so.0`, `libglib-2.0.so.0`,
  `libgthread-2.0.so.0`). Those come from OS packages -- `libgl1`, `libglx0`,
  `libglvnd0`, `libglib2.0-0` on Debian/Ubuntu -- which **no pip freeze can
  express**. On an image that lacks them, `import cv2` fails at load time with
  `ImportError: libGL.so.1: cannot open shared object file`, and this repository
  imports `cv2` unconditionally from `load_blender.py` and `load_LINEMOD.py`,
  both of which `run_nerf.py` imports at module scope. So the failure is at
  import of the training script, before any argument is parsed.

This script writes that boundary into the run's own output so the record carries
it rather than leaving a future reader to rediscover it. It is a diagnostic: it
prints and never fails the run.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def report_module(modname: str) -> None:
    print(f"--- {modname} ---")
    try:
        mod = __import__(modname)
    except Exception as exc:  # noqa: BLE001 - diagnostic
        print(f"  IMPORT FAILED: {type(exc).__name__}: {exc}")
        return
    version = getattr(mod, "__version__", "?")
    origin = getattr(mod, "__file__", "?")
    print(f"  version: {version}")
    print(f"  file:    {origin}")

    pkg_dir = Path(origin).parent if origin != "?" else None
    if pkg_dir is None:
        return
    sos = sorted(pkg_dir.glob("*.so")) + sorted(pkg_dir.glob("*.so.*"))
    for so in sos[:4]:
        print(f"  ldd {so.name}:")
        try:
            out = subprocess.run(
                ["ldd", str(so)], capture_output=True, text=True, timeout=60
            ).stdout
        except Exception as exc:  # noqa: BLE001 - diagnostic
            print(f"    ldd unavailable: {exc}")
            continue
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            # Anything resolved outside the wheel's own vendored lib dir is an
            # OS-package dependency the freeze cannot carry.
            vendored = ".libs/" in line or "site-packages" in line or "dist-packages" in line
            if "not found" in line:
                print(f"    MISSING  {line}")
            elif not vendored and "=>" in line:
                print(f"    OS-PKG   {line}")


def main() -> int:
    print(f"python: {sys.version.split()[0]} ({sys.executable})")
    print(f"platform: {' '.join(os.uname())}")

    report_module("cv2")
    report_module("imageio")

    print("--- imageio_ffmpeg ---")
    try:
        import imageio_ffmpeg

        print(f"  version: {imageio_ffmpeg.__version__}")
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"  ffmpeg exe: {exe}")
        inside_wheel = "site-packages" in exe or "dist-packages" in exe
        print(f"  bundled inside the wheel (pip-closed): {inside_wheel}")
        ver = subprocess.run(
            [exe, "-version"], capture_output=True, text=True, timeout=60
        ).stdout.splitlines()
        if ver:
            print(f"  {ver[0]}")
    except Exception as exc:  # noqa: BLE001 - diagnostic
        print(f"  IMPORT/EXEC FAILED: {type(exc).__name__}: {exc}")

    print("native_deps_probe: done (diagnostic only, never fatal)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
