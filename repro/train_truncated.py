#!/usr/bin/env python
"""Run upstream's `run_nerf.py` training loop, stopping after N iterations.

WHY THIS FILE EXISTS AT ALL
---------------------------
This repository's training length is **not configurable**. `run_nerf.py` sets

    N_iters = 200000 + 1                                   # run_nerf.py:701

as a local variable inside `train()`, and `config_parser()` exposes no flag for
it. Every other knob this row uses (`--i_weights`, `--i_testset`, `--i_video`,
`--config`) is an upstream argument passed on the command line; the iteration
count is the one thing that cannot be. A full 200,000-iteration run of the
`lego` config is measured in hours of single-GPU time, so a truncated
pipeline-viability run is the only affordable shape -- and truncating it needs
*something*.

There were two ways to get it, and this is the less invasive one:

  (a) add an `--N_iters` argument to `run_nerf.py`. That edits upstream's
      training script, which forfeits the property that makes this row worth
      anything: that not one line of the published model code changed.
  (b) leave `run_nerf.py` byte-identical and cap the iterator it loops over.

This file is (b). `run_nerf.py` iterates `for i in trange(start, N_iters)`,
where `trange` is a module-level name bound by `from tqdm import tqdm, trange`.
Rebinding `run_nerf.trange` to a version that shortens its stop value changes
how many times upstream's loop body runs and **nothing else** -- the model, the
sampler, the loss, the optimiser, the learning-rate schedule, the checkpoint
writer and the logging are all upstream's, unmodified, and they see exactly the
arguments upstream's own parser produces.

`repro/assert_upstream_unmodified.py` proves the claim mechanically: every
tracked file that came from upstream is compared, by git hash, against the
upstream commit this fork was taken from.

WHAT IS COPIED FROM UPSTREAM, AND WHY
-------------------------------------
`run_nerf.py`'s entry point is

    if __name__=='__main__':
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        train()

Entering through this wrapper means `run_nerf.__name__ != "__main__"`, so that
block does not execute. The `set_default_tensor_type` call is not incidental:
`raw2outputs()` and `render_rays()` build bare `torch.Tensor(...)` /
`torch.linspace(...)` values with no device argument and concatenate them with
GPU tensors, so without it the run dies on a device mismatch in the first
iteration. It is therefore reproduced here verbatim, and nowhere else does this
file touch torch.

TRUNCATION IS RECORDED, NOT HIDDEN
----------------------------------
`--train-iters` is part of the recorded argv, so the lineage states the
truncation on its face; a reader who wants the untruncated run raises that one
number. Nothing is set through the environment.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
# `python repro/train_truncated.py` puts repro/ on sys.path, not the repo root,
# so upstream's top-level modules (run_nerf, run_nerf_helpers, load_blender, ...)
# would not resolve. Doing it here, in committed code, keeps the requirement
# inside the recorded command instead of in a PYTHONPATH set around it -- an
# environment variable set outside `roar run` is not part of the lineage and is
# not replayed on a rebuild.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False, description=__doc__)
    ap.add_argument(
        "--train-iters",
        type=int,
        required=True,
        help="stop after this many optimiser iterations (upstream: 200000)",
    )
    ap.add_argument("--help", action="help")
    cfg, upstream_argv = ap.parse_known_args()

    if cfg.train_iters < 1:
        print("--train-iters must be >= 1")
        return 2

    # Hand upstream's parser exactly the arguments it expects, under the name it
    # expects. configargparse reads sys.argv[1:] and rejects unknown flags, so
    # --train-iters must not survive into it.
    sys.argv = ["run_nerf.py"] + upstream_argv

    import torch  # noqa: E402  (after the sys.path fix-up above)

    import run_nerf  # noqa: E402

    real_trange = run_nerf.trange

    def capped_trange(start, stop=None, *args, **kwargs):
        if stop is None:
            start, stop = 0, start
        limited = min(stop, start + cfg.train_iters)
        print(
            f"[truncation] upstream loop would run {start}..{stop - 1} "
            f"({stop - start} iterations); running {start}..{limited - 1} "
            f"({limited - start} iterations)",
            flush=True,
        )
        return real_trange(start, limited, *args, **kwargs)

    run_nerf.trange = capped_trange

    # Upstream's own __main__ line -- see the module docstring.
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    print(f"[truncation] torch {torch.__version__}, cuda available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[truncation] device: {torch.cuda.get_device_name(0)}", flush=True)

    t0 = time.time()
    run_nerf.train()
    print(f"[truncation] train() returned after {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
