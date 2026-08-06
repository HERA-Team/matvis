#!/usr/bin/env bash
# Canonical benchmark configs for tracking matvis GPU performance.
#
# Usage: profiling/run-canonical.sh [outdir] [dev|prodslice|both]
#
# Both configs are polarized with gridded (interpolated) beams and one unique
# beam per antenna, single precision — the production configuration.
#  * dev:       small enough to iterate quickly, still steady-state dominated.
#  * prodslice: 350 antennas / 350 beams, the production bottleneck ordering.
#               Fits a 4 GB GPU via auto-chunking.
set -euo pipefail

outdir="${1:-profiling/results}"
which="${2:-both}"
mkdir -p "$outdir"

common=(--gpu --interpolated-beam --single-precision --gpu-event-timing
        --coord-method CoordinateRotationERFA -f 1 -o "$outdir")

if [[ "$which" == "dev" || "$which" == "both" ]]; then
    uv run matvis profile -a 64 -b 64 -s 200000 -t 8 "${common[@]}"
fi

if [[ "$which" == "prodslice" || "$which" == "both" ]]; then
    # --nchunks 30 keeps the chunk size (~33k sources) identical on any GPU,
    # so results are comparable across machines.
    uv run matvis profile -a 350 -b 350 -s 1000000 -t 4 --nchunks 30 "${common[@]}"
fi
