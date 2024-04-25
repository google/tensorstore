load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

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
    name = "heterogeneous_container",
    hdrs = ["heterogeneous_container.h"],
    deps = ["@com_google_absl//absl/container:flat_hash_set"],
)

tensorstore_cc_test(
    name = "heterogeneous_container_test",
    size = "small",
    srcs = ["heterogeneous_container_test.cc"],
    deps = [
        ":heterogeneous_container",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "intrusive_linked_list",
    hdrs = ["intrusive_linked_list.h"],
)

tensorstore_cc_test(
    name = "intrusive_linked_list_test",
    size = "small",
    srcs = ["intrusive_linked_list_test.cc"],
    deps = [
        ":intrusive_linked_list",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "intrusive_red_black_tree",
    srcs = ["intrusive_red_black_tree.cc"],
    hdrs = ["intrusive_red_black_tree.h"],
    deps = [
        "//tensorstore/internal:tagged_ptr",
    ],
)

tensorstore_cc_test(
    name = "intrusive_red_black_tree_test",
    size = "small",
    srcs = ["intrusive_red_black_tree_test.cc"],
    deps = [
        ":intrusive_red_black_tree",
        "@com_google_absl//absl/random",
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
    name = "hash_set_of_any",
    hdrs = ["hash_set_of_any.h"],
    deps = ["@com_google_absl//absl/container:flat_hash_set"],
)

tensorstore_cc_test(
    name = "hash_set_of_any_test",
    srcs = ["hash_set_of_any_test.cc"],
    deps = [
        ":hash_set_of_any",
        "@com_google_googletest//:gtest_main",
    ],
)
