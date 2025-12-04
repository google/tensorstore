package(default_visibility = ["//visibility:public"])

# Minimal shim so //tensorstore/driver/netcdf:driver can depend on it.
# We just expose the system NetCDF library (-lnetcdf). Headers come from the
# system include path (libnetcdf-dev provides <netcdf.h>).
cc_library(
    name = "netcdf_util",
    hdrs = [],
    copts = [],
    linkopts = ["-lnetcdf"],
)
