load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

tensorstore_cc_library(
    name = "item_traits",
    hdrs = ["item_traits.h"],
    visibility = ["//visibility:private"],
    deps = ["@abseil-cpp//absl/meta:type_traits"],
)

tensorstore_cc_library(
    name = "block_queue",
    hdrs = ["block_queue.h"],
    deps = [
        ":item_traits",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:absl_check",
    ],
)

tensorstore_cc_test(
    name = "block_queue_test",
    srcs = ["block_queue_test.cc"],
    deps = [
        ":block_queue",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "circular_queue",
    hdrs = ["circular_queue.h"],
    deps = [
        ":item_traits",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:absl_check",
    ],
)

tensorstore_cc_test(
    name = "circular_queue_test",
    srcs = ["circular_queue_test.cc"],
    deps = [
        ":circular_queue",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "compressed_tuple",
    hdrs = ["compressed_tuple.h"],
)

tensorstore_cc_test(
    name = "compressed_tuple_test",
    srcs = ["compressed_tuple_test.cc"],
    deps = [
        ":compressed_tuple",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "heterogeneous_container",
    hdrs = ["heterogeneous_container.h"],
    deps = ["@abseil-cpp//absl/container:flat_hash_set"],
)

tensorstore_cc_test(
    name = "heterogeneous_container_test",
    size = "small",
    srcs = ["heterogeneous_container_test.cc"],
    deps = [
        ":heterogeneous_container",
        "@googletest//:gtest_main",
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
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "intrusive_red_black_tree",
    srcs = ["intrusive_red_black_tree.cc"],
    hdrs = ["intrusive_red_black_tree.h"],
    deps = [
        "//tensorstore/internal:tagged_ptr",
        "@abseil-cpp//absl/types:compare",
    ],
)

tensorstore_cc_test(
    name = "intrusive_red_black_tree_test",
    size = "small",
    srcs = ["intrusive_red_black_tree_test.cc"],
    deps = [
        ":intrusive_red_black_tree",
        "//tensorstore/internal:compare",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/types:compare",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "single_producer_queue",
    hdrs = ["single_producer_queue.h"],
    deps = [
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:absl_check",
    ],
)

tensorstore_cc_test(
    name = "single_producer_queue_test",
    srcs = ["single_producer_queue_test.cc"],
    deps = [
        ":single_producer_queue",
        "//tensorstore/internal/thread",
        "@abseil-cpp//absl/log:absl_check",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "hash_set_of_any",
    hdrs = ["hash_set_of_any.h"],
    deps = ["@abseil-cpp//absl/container:flat_hash_set"],
)

tensorstore_cc_test(
    name = "hash_set_of_any_test",
    srcs = ["hash_set_of_any_test.cc"],
    deps = [
        ":hash_set_of_any",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "multi_vector",
    hdrs = [
        "multi_vector.h",
        "multi_vector_impl.h",
    ],
    deps = [
        "//tensorstore:rank",
        "//tensorstore/internal:lldb_scripting",
        "//tensorstore/internal/meta",
        "//tensorstore/internal/meta:type_traits",
        "//tensorstore/util:division",
        "//tensorstore/util:span",
    ],
)

tensorstore_cc_test(
    name = "multi_vector_test",
    size = "small",
    srcs = ["multi_vector_test.cc"],
    deps = [
        ":multi_vector",
        "//tensorstore:rank",
        "//tensorstore/util:span",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "multi_vector_view",
    hdrs = ["multi_vector_view.h"],
    deps = [
        "//tensorstore:index",
        "//tensorstore:rank",
        "//tensorstore/internal:lldb_scripting",
        "//tensorstore/internal/meta",
        "//tensorstore/internal/meta:type_traits",
        "//tensorstore/util:span",
    ],
)

tensorstore_cc_test(
    name = "multi_vector_view_test",
    size = "small",
    srcs = ["multi_vector_view_test.cc"],
    deps = [
        ":multi_vector_view",
        "//tensorstore:rank",
        "//tensorstore/util:span",
        "@googletest//:gtest_main",
    ],
)
