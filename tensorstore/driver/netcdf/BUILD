package(default_visibility = ["//visibility:public"])

cc_library(
    name = "reader",
    hdrs = ["reader.h"],
    srcs = ["reader.cc"],
    linkopts = ["-lnetcdf"],
)

cc_library(
    name = "minidriver",
    hdrs = ["minidriver.h"],
    srcs = ["minidriver.cc"],
    linkopts = ["-lnetcdf"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "driver",
    hdrs = ["driver.h"],
    srcs = ["driver.cc"],
    deps = [
    "//tensorstore/driver/netcdf_util:netcdf_util",
        "//tensorstore/driver/netcdf:minidriver",
    ],
)


# Ensure the main driver target depends on the helpers (adjust if your driver target has a different name)
# If your driver target is already defined above, add "//tensorstore/driver/netcdf_util:netcdf_util" to its deps list.
