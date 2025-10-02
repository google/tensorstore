load(
    "//bazel:tensorstore.bzl",
    "tensorstore_cc_library",
    "tensorstore_cc_test",
)

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

tensorstore_cc_library(
    name = "concurrent",
    testonly = 1,
    srcs = ["concurrent.cc"],
    hdrs = ["concurrent.h"],
    deps = [
        "//tensorstore/internal:multi_barrier",
        "//tensorstore/internal/thread",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
    ],
)

tensorstore_cc_test(
    name = "concurrent_test",
    srcs = ["concurrent_test.cc"],
    deps = [
        ":concurrent",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/synchronization",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "decoded_matches",
    testonly = 1,
    srcs = ["decoded_matches.cc"],
    hdrs = ["decoded_matches.h"],
    deps = [
        "//tensorstore/util:result",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@googletest//:gtest",
    ],
)

tensorstore_cc_test(
    name = "decoded_matches_test",
    size = "small",
    srcs = ["decoded_matches_test.cc"],
    deps = [
        ":decoded_matches",
        "//tensorstore/util:result",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "dynamic",
    testonly = 1,
    srcs = ["dynamic.cc"],
    hdrs = ["dynamic.h"],
    deps = [
        "//tensorstore/internal:source_location",
        "@googletest//:gtest",
    ],
)

tensorstore_cc_library(
    name = "half_gtest",
    testonly = 1,
    hdrs = ["half_gtest.h"],
    deps = ["@net_sourceforge_half//:half"],
)

tensorstore_cc_library(
    name = "json_gtest",
    testonly = 1,
    srcs = ["json_gtest.cc"],
    hdrs = ["json_gtest.h"],
    deps = [
        "//tensorstore/internal:json_pointer",
        "//tensorstore/internal/json:same",
        "//tensorstore/util:quote_string",
        "@googletest//:gtest",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_test(
    name = "json_gtest_test",
    size = "small",
    srcs = ["json_gtest_test.cc"],
    deps = [
        ":json_gtest",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "non_compile_bypass",
    testonly = True,
    hdrs = ["non_compile_bypass.h"],
)

tensorstore_cc_library(
    name = "parse_json_matches",
    testonly = 1,
    srcs = ["parse_json_matches.cc"],
    hdrs = ["parse_json_matches.h"],
    deps = [
        ":json_gtest",
        "//tensorstore/internal/json_binding",
        "@googletest//:gtest",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_test(
    name = "parse_json_matches_test",
    size = "small",
    srcs = ["parse_json_matches_test.cc"],
    deps = [
        ":parse_json_matches",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "queue_testutil",
    testonly = 1,
    hdrs = ["queue_testutil.h"],
    deps = [
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
    ],
)

tensorstore_cc_library(
    name = "random_seed",
    testonly = 1,
    srcs = ["random_seed.cc"],
    hdrs = ["random_seed.h"],
    deps = [
        "//tensorstore/internal:env",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/strings",
    ],
)

tensorstore_cc_library(
    name = "scoped_directory",
    testonly = 1,
    srcs = ["scoped_directory.cc"],
    hdrs = ["scoped_directory.h"],
    deps = [
        "//tensorstore/internal:path",
        "//tensorstore/internal/os:cwd",
        "//tensorstore/internal/os:file_util",
        "//tensorstore/internal/os:filesystem",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
    ],
)

tensorstore_cc_library(
    name = "on_windows",
    testonly = 1,
    hdrs = ["on_windows.h"],
    deps = ["@abseil-cpp//absl/base:core_headers"],
)
