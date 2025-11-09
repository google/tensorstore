load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

tensorstore_cc_library(
    name = "cache_key",
    hdrs = [
        "absl_time.h",
        "cache_key.h",
        "fwd.h",
        "std_optional.h",
        "std_variant.h",
        "std_vector.h",
    ],
    deps = [
        "//tensorstore/util/apply_members",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/time",
    ],
)

tensorstore_cc_test(
    name = "cache_key_test",
    srcs = ["cache_key_test.cc"],
    deps = [
        ":cache_key",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "json",
    hdrs = ["json.h"],
    deps = [
        ":cache_key",
        "@nlohmann_json//:json",
    ],
)
