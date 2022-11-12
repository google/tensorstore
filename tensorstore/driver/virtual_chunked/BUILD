load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# This source file is compiled as part of the
# //tensorstore:virtual_chunked target in order to allow the
# `virtual_chunked.h` header to be part of the top-level package.
exports_files(["virtual_chunked.cc"])

# Simple forwarding target.
tensorstore_cc_library(
    name = "virtual_chunked",
    deps = ["//tensorstore:virtual_chunked"],
)

filegroup(
    name = "doc_sources",
    srcs = [],
)

tensorstore_cc_test(
    name = "virtual_chunked_test",
    srcs = ["virtual_chunked_test.cc"],
    deps = [
        "//tensorstore:virtual_chunked",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/internal:queue_testutil",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:generation_testutil",
        "//tensorstore/serialization",
        "//tensorstore/serialization:test_util",
        "//tensorstore/util:iterate_over_index_range",
        "//tensorstore/util:span",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
    ],
)
