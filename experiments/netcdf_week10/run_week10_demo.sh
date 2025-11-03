#!/usr/bin/env bash
set -euo pipefail
bazel build //experiments/netcdf_week10:write_demo //experiments/netcdf_week10:read_demo //experiments/netcdf_week10:roundtrip_test
./bazel-bin/experiments/netcdf_week10/write_demo week10_out.nc
./bazel-bin/experiments/netcdf_week10/read_demo week10_out.nc
./bazel-bin/experiments/netcdf_week10/roundtrip_test
