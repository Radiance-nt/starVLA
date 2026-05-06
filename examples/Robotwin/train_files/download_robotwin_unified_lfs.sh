#!/bin/bash

set -euo pipefail

DATASET_DIR="${DATASET_DIR:-$HOME/datasets/robotwin_unified}"
LOG_DIR="${LOG_DIR:-$DATASET_DIR/.download_logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/git_lfs_pull_$(date +%Y%m%d_%H%M%S).log}"
PID_FILE="${PID_FILE:-$LOG_DIR/git_lfs_pull.pid}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-60}"
STARVLA_PYTHON="${STARVLA_PYTHON:-/mnt/petrelfs/linzhanhui/miniconda3/envs/starVLA/bin/python}"

mkdir -p "$LOG_DIR"

if [[ ! -d "$DATASET_DIR/.git" ]]; then
  echo "Dataset repo not found: $DATASET_DIR" >&2
  exit 1
fi

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${existing_pid}" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "A download is already running with PID $existing_pid" >&2
    echo "Log: $LOG_FILE" >&2
    exit 1
  fi
  rm -f "$PID_FILE"
fi

count_tracked_files() {
  git lfs ls-files | wc -l | awk '{print $1}'
}

count_materialized_files() {
  local count=0
  while IFS= read -r path; do
    [[ -f "$path" ]] || continue
    if ! head -n 1 "$path" | grep -q '^version https://git-lfs.github.com/spec/v1$'; then
      count=$((count + 1))
    fi
  done < <(git lfs ls-files | awk '{print $NF}')
  echo "$count"
}

count_pointer_files() {
  local count=0
  while IFS= read -r path; do
    [[ -f "$path" ]] || continue
    if head -n 1 "$path" | grep -q '^version https://git-lfs.github.com/spec/v1$'; then
      count=$((count + 1))
    fi
  done < <(git lfs ls-files | awk '{print $NF}')
  echo "$count"
}

exec > >(tee -a "$LOG_FILE") 2>&1

cd "$DATASET_DIR"

echo "[INFO] Start time: $(date --iso-8601=seconds)"
echo "[INFO] Dataset dir: $DATASET_DIR"
echo "[INFO] Log file: $LOG_FILE"
echo "[INFO] PID file: $PID_FILE"
echo "[INFO] Tracked LFS files: $(count_tracked_files)"

git lfs pull &
pull_pid=$!
echo "$pull_pid" > "$PID_FILE"
echo "[INFO] git lfs pull pid: $pull_pid"

while kill -0 "$pull_pid" 2>/dev/null; do
  materialized="$(count_materialized_files)"
  tracked="$(count_tracked_files)"
  pointers="$(count_pointer_files)"
  echo "[PROGRESS] $(date --iso-8601=seconds) materialized=${materialized}/${tracked} pointers_remaining=${pointers}"
  sleep "$PROGRESS_INTERVAL"
done

wait "$pull_pid"
rm -f "$PID_FILE"

echo "[INFO] git lfs pull finished at $(date --iso-8601=seconds)"
echo "[INFO] Running git lfs fsck"
git lfs fsck

pointers="$(count_pointer_files)"
echo "[INFO] Pointer files remaining after pull: $pointers"
if [[ "$pointers" != "0" ]]; then
  echo "[ERROR] Some LFS pointer files remain in the working tree." >&2
  exit 1
fi

echo "[INFO] Validating parquet readability"
DATASET_DIR="$DATASET_DIR" "$STARVLA_PYTHON" - <<'PY'
import os
from pathlib import Path
import pyarrow.parquet as pq

root = Path(os.environ["DATASET_DIR"])
data_files = sorted((root / "data").glob("chunk-*/*.parquet"))
episode_files = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))

if not data_files:
    raise SystemExit("No data parquet files found")
if not episode_files:
    raise SystemExit("No episode parquet files found")

for path in data_files:
    pq.read_metadata(path)
for path in episode_files:
    pq.read_metadata(path)

print(f"[INFO] Readable data parquet files: {len(data_files)}")
print(f"[INFO] Readable episode parquet files: {len(episode_files)}")
PY

echo "[INFO] Validation passed at $(date --iso-8601=seconds)"
