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
    name = "decoded_matches",
    testonly = 1,
    srcs = ["decoded_matches.cc"],
    hdrs = ["decoded_matches.h"],
    deps = [
        "//tensorstore/util:result",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest",
    ],
)

tensorstore_cc_test(
    name = "decoded_matches_test",
    size = "small",
    srcs = ["decoded_matches_test.cc"],
    deps = [
        ":decoded_matches",
        "//tensorstore/util:result",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "dynamic",
    testonly = 1,
    srcs = ["dynamic.cc"],
    hdrs = ["dynamic.h"],
    deps = [
        "//tensorstore/internal:source_location",
        "@com_google_googletest//:gtest",
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
        "@com_github_nlohmann_json//:json",
        "@com_google_googletest//:gtest",
    ],
)

tensorstore_cc_test(
    name = "json_gtest_test",
    size = "small",
    srcs = ["json_gtest_test.cc"],
    deps = [
        ":json_gtest",
        "@com_github_nlohmann_json//:json",
        "@com_google_googletest//:gtest_main",
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
        "@com_github_nlohmann_json//:json",
        "@com_google_googletest//:gtest",
    ],
)

tensorstore_cc_test(
    name = "parse_json_matches_test",
    size = "small",
    srcs = ["parse_json_matches_test.cc"],
    deps = [
        ":parse_json_matches",
        "@com_github_nlohmann_json//:json",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "queue_testutil",
    testonly = 1,
    hdrs = ["queue_testutil.h"],
    deps = [
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_library(
    name = "random_seed",
    testonly = 1,
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
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)
