#!/usr/bin/env bash
set -euo pipefail

# 1) Build a tiny dataset
cat > /tmp/week8_minimal.cdl <<'EOF'
netcdf week8_minimal {
  dimensions: time = 6 ; lat = 2 ; lon = 3 ;
  variables: int time(time) ; float temp(time) ; double grid(lat, lon) ;
  data:
    time = 0,1,2,3,4,5 ;
    temp = 20.5,21.0,19.8,18.9,22.2,23.1 ;
    grid = 1.0,1.1,1.2,
           2.0,2.1,2.2 ;
}
EOF
ncgen -o /tmp/week8_minimal.nc /tmp/week8_minimal.cdl

# 2) Build the reader (Homebrew nc-config assumed)
eval "$(/opt/homebrew/bin/brew shellenv)" || true
clang++ -std=c++17 experiments/netcdf_week8/ts_netcdf_read.cc -o /tmp/ts_netcdf_read \
  $(nc-config --cflags) $(nc-config --libs)

# 3) Run demos
echo "== temp[:] ==";        /tmp/ts_netcdf_read /tmp/week8_minimal.nc temp
echo "== temp[1:5] ==";      /tmp/ts_netcdf_read /tmp/week8_minimal.nc temp '1:5'
echo "== grid[:,:] ==";      /tmp/ts_netcdf_read /tmp/week8_minimal.nc grid ':,:'
echo "== grid[:,1:3] ==";    /tmp/ts_netcdf_read /tmp/week8_minimal.nc grid '0:2,1:3'
echo "== grid[:,::2] ==";    /tmp/ts_netcdf_read /tmp/week8_minimal.nc grid ':,::2'
