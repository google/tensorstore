load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

tensorstore_cc_library(
    name = "ascii_set",
    hdrs = ["ascii_set.h"],
)

tensorstore_cc_library(
    name = "percent_coder",
    srcs = ["percent_coder.cc"],
    hdrs = ["percent_coder.h"],
    deps = [
        ":ascii_set",
        "//tensorstore/internal:utf8",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
    ],
)

tensorstore_cc_test(
    name = "percent_coder_test",
    srcs = ["percent_coder_test.cc"],
    deps = [
        ":ascii_set",
        ":percent_coder",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "path",
    srcs = ["path.cc"],
    hdrs = ["path.h"],
    deps = [
        ":parse",
        ":percent_coder",
        "//tensorstore/internal:path",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
    ],
)

tensorstore_cc_library(
    name = "parse",
    srcs = ["parse.cc"],
    hdrs = ["parse.h"],
    deps = [
        ":ascii_set",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
    ],
)

tensorstore_cc_test(
    name = "path_test",
    srcs = ["path_test.cc"],
    deps = [
        ":parse",
        ":path",
        "//tensorstore/internal/testing:on_windows",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "parse_test",
    srcs = ["parse_test.cc"],
    deps = [
        ":parse",
        "@googletest//:gtest_main",
    ],
)
