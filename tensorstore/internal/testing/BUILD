load(
    "//bazel:tensorstore.bzl",
    "tensorstore_cc_library",
    "tensorstore_cc_test",
)

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

tensorstore_cc_library(
    name = "concurrent",
    testonly = True,
    srcs = ["concurrent.cc"],
    hdrs = ["concurrent.h"],
    deps = [
        "//tensorstore/internal:multi_barrier",
        "//tensorstore/internal/thread",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
    ],
)

tensorstore_cc_test(
    name = "concurrent_test",
    srcs = ["concurrent_test.cc"],
    deps = [
        ":concurrent",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "dynamic",
    testonly = True,
    srcs = ["dynamic.cc"],
    hdrs = ["dynamic.h"],
    deps = [
        "//tensorstore/internal:source_location",
        "@com_google_googletest//:gtest",
    ],
)

tensorstore_cc_library(
    name = "random_seed",
    testonly = True,
    srcs = ["random_seed.cc"],
    hdrs = ["random_seed.h"],
    deps = [
        "//tensorstore/internal:env",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_library(
    name = "scoped_directory",
    testonly = True,
    srcs = ["scoped_directory.cc"],
    hdrs = ["scoped_directory.h"],
    deps = [
        "//tensorstore/internal:path",
        "//tensorstore/internal/os:filesystem",
        "//tensorstore/kvstore/file:file_util",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/random",
    ],
)
