#!/usr/bin/env bash
# Copyright (c) 2026 Emir Özer. SPDX-License-Identifier: BSD-3-Clause
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 PREPARED_DATASET VIDEO_DIRECTORY NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dataset="$(realpath -e -- "$1")"
videos="$(realpath -e -- "$2")"
output="$(realpath -m -- "$3")"
binary="${PVO_BINARY:-$repo_root/build-locked/basalt_vio}"
binary="$(realpath -e -- "$binary")"
[[ -x "$binary" ]] || { echo "Estimator is not executable: $binary" >&2; exit 2; }
for result in tracking.csv result.json; do
  [[ ! -e "$output/$result" ]] || { echo "Result already exists: $output/$result" >&2; exit 2; }
done
mkdir -p -- "$output"
cd -- "$output"
exec "$binary" \
  --show-gui 0 --deterministic 1 --num-threads 1 \
  --dataset-path "$dataset" --video-dataset-path "$videos" \
  --dataset-type euroc \
  --cam-calib "$repo_root/data/euroc/euroc_ds_calib.json" \
  --config-path "$repo_root/configs/pvo.json" \
  --use-mvs 1 --use-video-frames 1 --use-imu 0 \
  --save-trajectory euroc --save-trajectory-fn tracking.csv \
  --result-path result.json \
  --direct-mv-stereo-reprojection-gate 1.5 \
  --direct-mv-teaser-noise-bound-m 0.1 \
  --direct-mv-teaser-far-depth-m 5 \
  --direct-mv-teaser-far-reprojection-gate-px 1.5 \
  --temporal-reverse-policy hybrid \
  --temporal-reverse-cosine-threshold 0.98 \
  --temporal-reverse-min-iou 0.15 \
  --temporal-reverse-max-cycle-error 1.5 \
  --direct-mv-merge-indistinguishable \
  --direct-mv-merge-radius-px 1.125
