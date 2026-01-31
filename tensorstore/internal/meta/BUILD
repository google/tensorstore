load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

tensorstore_cc_library(
    name = "attributes",
    hdrs = ["attributes.h"],
    deps = ["@abseil-cpp//absl/base:core_headers"],
)

tensorstore_cc_library(
    name = "exception_macros",
    hdrs = ["exception_macros.h"],
    deps = [
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:absl_check",
    ],
)

tensorstore_cc_library(
    name = "integer_range",
    hdrs = ["integer_range.h"],
)

tensorstore_cc_library(
    name = "integer_sequence",
    hdrs = ["integer_sequence.h"],
)

tensorstore_cc_library(
    name = "integer_types",
    hdrs = ["integer_types.h"],
)

tensorstore_cc_library(
    name = "meta",
    hdrs = ["meta.h"],
)

tensorstore_cc_test(
    name = "meta_test",
    size = "small",
    srcs = ["meta_test.cc"],
    deps = [
        ":meta",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "requires",
    hdrs = ["requires.h"],
)

tensorstore_cc_library(
    name = "type_traits",
    hdrs = ["type_traits.h"],
    deps = [
        "//tensorstore:index",
        "@abseil-cpp//absl/meta:type_traits",
    ],
)

tensorstore_cc_test(
    name = "type_traits_test",
    size = "small",
    srcs = ["type_traits_test.cc"],
    deps = [
        ":type_traits",
        "@googletest//:gtest_main",  # buildcleaner: keep
    ],
)

tensorstore_cc_library(
    name = "void_wrapper",
    hdrs = ["void_wrapper.h"],
    deps = ["@abseil-cpp//absl/status"],
)

tensorstore_cc_test(
    name = "void_wrapper_test",
    size = "small",
    srcs = ["void_wrapper_test.cc"],
    deps = [
        ":type_traits",
        ":void_wrapper",
        "@googletest//:gtest_main",
    ],
)
