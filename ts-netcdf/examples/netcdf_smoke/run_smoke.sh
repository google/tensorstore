#!/usr/bin/env bash
set -euo pipefail
# Bazel will stage the binary at this runfile path:
BIN="examples/netcdf_smoke/open_netcdf_smoke"
# Run it and test its output:
OUT="$($BIN)"
echo "$OUT"
grep -q "netCDF driver linked. Smoke OK." <<<"$OUT" || {
  echo "Expected output not found"
  exit 1
}
