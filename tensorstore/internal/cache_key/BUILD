load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(
    default_visibility = ["//tensorstore:internal_packages"],
)

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
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_test(
    name = "cache_key_test",
    srcs = ["cache_key_test.cc"],
    deps = [
        ":cache_key",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)
