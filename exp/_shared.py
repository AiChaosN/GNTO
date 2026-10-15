"""Small helpers shared by every experiment directory (no experiment-specific logic)."""
from __future__ import annotations

import os
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

GNTO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class Logger:
    """print + append to ``<path>`` with a timestamp."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a")

    def __call__(self, msg: str) -> None:
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_rev() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=GNTO_PATH,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def provenance() -> dict:
    """Fingerprint the actual source tree, including uncommitted/new experiment code.

    An old HEAD alone cannot identify experiments in a dirty worktree. No environment
    variables or database credentials are captured here.
    """
    root = Path(GNTO_PATH)
    paths = [p for directory in ("models", "adapters", "exp", "scripts")
             for p in (root / directory).rglob("*.py") if "__pycache__" not in p.parts]
    paths += [p for p in root.glob("requirements*.txt")]
    hashes = {str(p.relative_to(root)): file_sha256(p) for p in sorted(paths)}
    external = {}
    for repo in ("BaoForPostgreSQL", "LIMAOLifeLongRLDB"):
        folder = root.parent / repo / "bao_server"
        files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix in (".py", ".pkl"))
        external[repo] = {str(p.relative_to(root.parent / repo)): file_sha256(p) for p in files}
    versions = {}
    for name in ("numpy", "pandas", "torch", "torch-geometric", "scipy", "scikit-learn",
                 "matplotlib", "loralib", "kmodes", "psycopg2-binary"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"git_rev": git_rev(), "source_sha256": object_sha256({"local": hashes, "external": external}),
            "source_files": hashes, "external_source_files": external,
            "python": platform.python_version(), "platform": platform.platform(), "packages": versions}


def md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """DataFrame -> Markdown table."""
    if df is None or df.empty:
        return "_(empty)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (bool, np.bool_)):
                cells.append(str(v))
            elif isinstance(v, (float, np.floating)):
                cells.append(floatfmt.format(v))
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{int(v):,}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
