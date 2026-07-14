#!/bin/bash

set -euo pipefail

HOME_DIR="${HOME:-/mnt/petrelfs/linzhanhui}"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY
else
  export WANDB_API_KEY="$(
    python3 - <<'PY'
from netrc import netrc
from pathlib import Path
import os
import sys

home = Path(os.environ.get("HOME", "/mnt/petrelfs/linzhanhui"))
auth = netrc(str(home / ".netrc")).authenticators("api.wandb.ai")
if not auth or not auth[2]:
    raise SystemExit("Missing wandb token in ~/.netrc")
sys.stdout.write(auth[2])
PY
  )"
fi
