#!/usr/bin/env python
"""Prove, mechanically, that no upstream file in this fork has been modified.

The reproduction claim this row makes is "upstream's training script, upstream's
config, upstream's data loaders, run as published". `repro/train_truncated.py`
caps the iteration count by rebinding a module attribute rather than by editing
`run_nerf.py`, precisely so that claim stays literally true -- but a prose claim
in a docstring is worth nothing without a check that fails when it stops being
true.

This compares every path in the upstream commit against the same path in HEAD by
git blob hash. Any upstream file that differs, or is missing, is a failure. Files
that exist only in HEAD are reported as additions (this fork's `repro/` helpers,
its `.treqs/` workflow and its `.gitkeep` markers) and are allowed: they are new
files, not modifications of published code.

One upstream file is knowingly modified and must be declared with
`--allow-modified`: **`.gitignore`**. Upstream already ignores `data/*` and
`logs/*`; this fork adds `!data/.gitkeep`, `!logs/.gitkeep` and the nested
`logs/blender_paper_lego/` rules so those output directories survive a clean
checkout with a tracked marker file. Without that, a rebuilt worktree has no
`logs/` at all and the training step's outputs fall out of the lineage. It is
repository plumbing, contains no code, and is printed in full by this script so
the exemption cannot be quiet. Nothing under `run_nerf.py`, `run_nerf_helpers.py`,
`load_*.py`, `configs/` or `requirements.txt` is exempt.

Run it in the untraced setup stage, before anything is paid for.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

UPSTREAM_COMMIT = "63a5a630c9abd62b0f21c08703d0ac2ea7d4b9dd"
UPSTREAM_REPO = "https://github.com/yenchenlin/nerf-pytorch"


def ensure_commit(commit: str) -> bool:
    """Make `commit` readable locally, fetching it if this is a shallow clone.

    A rebuild host may clone at a depth that does not include the upstream commit
    this fork branched from. Fetch just that object if so; if the network refuses,
    say loudly that the check could not run rather than aborting a run over it.
    """
    have = subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"]).returncode == 0
    if have:
        return True
    print(f"upstream commit {commit[:12]} not present locally; fetching it")
    fetched = subprocess.run(
        ["git", "fetch", "--no-tags", "--depth=1", UPSTREAM_REPO, commit],
        capture_output=True,
        text=True,
    )
    if fetched.returncode != 0:
        print(fetched.stderr.strip())
        return False
    return subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"]).returncode == 0


def ls_tree(ref: str) -> dict[str, str]:
    out = subprocess.run(
        ["git", "ls-tree", "-r", ref],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tree: dict[str, str] = {}
    for line in out.splitlines():
        meta, path = line.split("\t", 1)
        _mode, _type, blob = meta.split()
        tree[path] = blob
    return tree


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--upstream-commit", default=UPSTREAM_COMMIT)
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument(
        "--allow-modified",
        action="append",
        default=[],
        help="upstream path that may differ (repeatable). Its full diff is printed.",
    )
    cfg = ap.parse_args()

    print(f"upstream: {UPSTREAM_REPO} @ {cfg.upstream_commit}")
    if not ensure_commit(cfg.upstream_commit):
        print("=" * 70)
        print("assert_upstream_unmodified: SKIPPED -- the upstream commit is not")
        print("reachable from this checkout and could not be fetched. The claim")
        print("that upstream's files are unmodified is NOT verified on this host.")
        print("=" * 70)
        return 0
    upstream = ls_tree(cfg.upstream_commit)
    head = ls_tree(cfg.ref)

    changed = sorted(p for p, b in upstream.items() if p in head and head[p] != b)
    allowed = set(cfg.allow_modified)
    modified = [p for p in changed if p not in allowed]
    declared = [p for p in changed if p in allowed]
    removed = sorted(p for p in upstream if p not in head)
    added = sorted(p for p in head if p not in upstream)

    print(f"upstream files: {len(upstream)}  |  files in {cfg.ref}: {len(head)}")
    print(f"added by this fork ({len(added)}):")
    for p in added:
        print(f"  + {p}")

    for p in declared:
        print(f"  M {p}   <-- DECLARED modification, diff follows:")
        diff = subprocess.run(
            ["git", "diff", f"{cfg.upstream_commit}..{cfg.ref}", "--", p],
            capture_output=True,
            text=True,
        ).stdout
        for line in diff.splitlines():
            print(f"      {line}")

    # An --allow-modified path that is NOT actually modified means the exemption
    # list has drifted from reality; say so, but do not fail on it.
    for p in sorted(allowed - set(changed)):
        print(f"  ? {p} declared with --allow-modified but is unchanged")

    if modified or removed:
        for p in modified:
            print(f"  M {p}   <-- UNDECLARED UPSTREAM MODIFICATION")
        for p in removed:
            print(f"  D {p}   <-- UPSTREAM FILE REMOVED")
        print("assert_upstream_unmodified: FAIL")
        return 1

    print(
        f"assert_upstream_unmodified: PASS "
        f"({len(upstream) - len(declared)} upstream files byte-identical to "
        f"{cfg.upstream_commit[:12]}, {len(declared)} declared exemption(s))"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
