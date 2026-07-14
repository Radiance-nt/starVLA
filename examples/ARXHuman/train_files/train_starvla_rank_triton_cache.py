#!/usr/bin/env python
"""Launch train_starvla.py with an isolated Triton cache per process."""

import os
import runpy
from pathlib import Path


def main() -> None:
    base = Path(os.environ.get("STARVLA_TRITON_CACHE_BASE", "/tmp/starvla_triton_cache"))
    job_id = os.environ.get("SLURM_JOB_ID", "no_slurm_job")
    node_rank = os.environ.get("SLURM_PROCID", "0")
    global_rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    local_rank = os.environ.get("LOCAL_RANK", "0")

    cache_dir = base / job_id / f"node{node_rank}_rank{global_rank}_local{local_rank}"
    (cache_dir / "autotune").mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)

    runpy.run_path("starVLA/training/train_starvla.py", run_name="__main__")


if __name__ == "__main__":
    main()
