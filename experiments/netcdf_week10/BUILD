load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_library(
    name = "ts_netcdf_io",
    srcs = ["ts_netcdf_io.cc"],
    hdrs = ["ts_netcdf_io.h"],
    copts = ["-std=gnu++17"],
    linkopts = ["-lnetcdf"],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "write_demo",
    srcs = ["write_demo.cc"],
    deps = [":ts_netcdf_io"],
    copts = ["-std=gnu++17"],
    linkopts = ["-lnetcdf"],
)

cc_binary(
    name = "read_demo",
    srcs = ["read_demo.cc"],
    deps = [":ts_netcdf_io"],
    copts = ["-std=gnu++17"],
    linkopts = ["-lnetcdf"],
)

cc_binary(
    name = "roundtrip_test",
    srcs = ["roundtrip_test.cc"],
    deps = [":ts_netcdf_io"],
    copts = ["-std=gnu++17"],
    linkopts = ["-lnetcdf"],
)

cc_binary(
    name = "comprehensive_write_test",
    srcs = ["comprehensive_write_test.cc"],
    deps = ["//ts-netcdf/tensorstore/driver/netcdf:minidriver_lib"],
    copts = ["-std=gnu++17"],
    linkopts = ["-lnetcdf"],
)
