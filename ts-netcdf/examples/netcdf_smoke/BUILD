cc_binary(
    name = "open_netcdf_smoke",
    srcs = ["open_netcdf_smoke.cc"],
    deps = [
        "//tensorstore/driver/netcdf:driver",  # pulls in the netCDF driver (registration)
        "//tensorstore:tensorstore",           # core TensorStore library
    ],
)

sh_test(
    name = "open_netcdf_smoke_test",
    srcs = ["run_smoke.sh"],
    data = [":open_netcdf_smoke"],
)
