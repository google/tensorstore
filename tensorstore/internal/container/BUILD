load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

tensorstore_cc_library(
    name = "circular_queue",
    hdrs = ["circular_queue.h"],
    deps = [
        "//tensorstore/internal:attributes",
        "@com_google_absl//absl/log:absl_check",
    ],
)

tensorstore_cc_test(
    name = "circular_queue_test",
    srcs = ["circular_queue_test.cc"],
    deps = [
        ":circular_queue",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "compressed_pair",
    hdrs = ["compressed_pair.h"],
)

tensorstore_cc_test(
    name = "compressed_pair_test",
    size = "small",
    srcs = ["compressed_pair_test.cc"],
    deps = [
        ":compressed_pair",
        "//tensorstore/internal:type_traits",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "single_producer_queue",
    hdrs = ["single_producer_queue.h"],
    deps = [
        "//tensorstore/internal:attributes",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_check",
    ],
)

tensorstore_cc_test(
    name = "single_producer_queue_test",
    srcs = ["single_producer_queue_test.cc"],
    deps = [
        ":single_producer_queue",
        "//tensorstore/internal/thread",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "block_queue",
    hdrs = ["block_queue.h"],
    deps = [
        "//tensorstore/internal:attributes",
        "@com_google_absl//absl/log:absl_check",
    ],
)

tensorstore_cc_test(
    name = "block_queue_test",
    srcs = ["block_queue_test.cc"],
    deps = [
        ":block_queue",
        "@com_google_googletest//:gtest_main",
    ],
)
