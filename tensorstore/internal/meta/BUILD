load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

tensorstore_cc_library(
    name = "attributes",
    hdrs = ["attributes.h"],
    deps = ["@com_google_absl//absl/base:core_headers"],
)

tensorstore_cc_library(
    name = "exception_macros",
    hdrs = ["exception_macros.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_check",
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
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "type_traits",
    hdrs = ["type_traits.h"],
    deps = [
        "//tensorstore:index",
        "@com_google_absl//absl/meta:type_traits",
    ],
)

tensorstore_cc_test(
    name = "type_traits_test",
    size = "small",
    srcs = ["type_traits_test.cc"],
    deps = [
        ":type_traits",
        "@com_google_googletest//:gtest_main",  # buildcleaner: keep
    ],
)

tensorstore_cc_library(
    name = "void_wrapper",
    hdrs = ["void_wrapper.h"],
    deps = ["@com_google_absl//absl/status"],
)

tensorstore_cc_test(
    name = "void_wrapper_test",
    size = "small",
    srcs = ["void_wrapper_test.cc"],
    deps = [
        ":type_traits",
        ":void_wrapper",
        "@com_google_googletest//:gtest_main",
    ],
)
