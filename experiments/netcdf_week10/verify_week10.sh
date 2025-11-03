#!/usr/bin/env bash
set -euo pipefail
bazel build //experiments/netcdf_week10:roundtrip_test
./bazel-bin/experiments/netcdf_week10/roundtrip_test
