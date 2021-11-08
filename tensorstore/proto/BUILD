load("//tensorstore:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_proto_library", "tensorstore_cc_test", "tensorstore_proto_library")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

tensorstore_proto_library(
    name = "array_proto",
    srcs = ["array.proto"],
)

tensorstore_cc_proto_library(
    name = "array_cc_proto",
    deps = [":array_proto"],
)

tensorstore_proto_library(
    name = "index_transform_proto",
    srcs = ["index_transform.proto"],
)

tensorstore_cc_proto_library(
    name = "index_transform_cc_proto",
    deps = [":index_transform_proto"],
)

tensorstore_proto_library(
    name = "schema_proto",
    srcs = ["schema.proto"],
    deps = [
        ":array_proto",
        ":index_transform_proto",
    ],
)

tensorstore_cc_proto_library(
    name = "schema_cc_proto",
    deps = [":schema_proto"],
)

tensorstore_cc_library(
    name = "array",
    srcs = ["array.cc"],
    hdrs = ["array.h"],
    deps = [
        ":array_cc_proto",
        "//tensorstore:array",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore:rank",
        "//tensorstore:strided_layout",
        "//tensorstore/internal:elementwise_function",
        "//tensorstore/internal:unaligned_data_type_functions",
        "//tensorstore/util:dimension_set",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@com_google_absl//absl/status",
        "@com_google_riegeli//riegeli/bytes:string_reader",
        "@com_google_riegeli//riegeli/bytes:string_writer",
    ],
)

tensorstore_cc_test(
    name = "array_test",
    srcs = ["array_test.cc"],
    deps = [
        ":array",
        ":protobuf_matchers",
        ":test_util",
        "//tensorstore:array_testutil",
        "//tensorstore:index",
        "//tensorstore:strided_layout",
        "//tensorstore/index_space:index_transform_testutil",
        "//tensorstore/internal:data_type_random_generator",
        "//tensorstore/internal:test_util",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "index_transform",
    srcs = ["index_transform.cc"],
    hdrs = ["index_transform.h"],
    deps = [
        ":index_transform_cc_proto",
        "//tensorstore:index",
        "//tensorstore:json_serialization_options",
        "//tensorstore:rank",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:dimension_indexed_json_binder",
        "//tensorstore/internal:json",
        "//tensorstore/internal:json_array",
        "//tensorstore/internal:logging",
        "//tensorstore/internal:type_traits",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_test(
    name = "index_transform_test",
    srcs = ["index_transform_test.cc"],
    deps = [
        ":index_transform",
        ":index_transform_cc_proto",
        ":protobuf_matchers",
        ":test_util",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "schema",
    srcs = ["schema.cc"],
    hdrs = ["schema.h"],
    deps = [
        ":array",
        ":index_transform",
        ":schema_cc_proto",
        "//tensorstore:array",
        "//tensorstore:chunk_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:rank",
        "//tensorstore:schema",
        "//tensorstore:strided_layout",
        "//tensorstore/index_space:dimension_units",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:logging",
        "//tensorstore/internal:type_traits",
        "//tensorstore/serialization",
        "//tensorstore/serialization:batch",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_test(
    name = "schema_test",
    srcs = ["schema_test.cc"],
    deps = [
        ":index_transform_cc_proto",
        ":protobuf_matchers",
        ":schema",
        ":test_util",
        "//tensorstore:schema",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "test_util",
    testonly = 1,
    hdrs = ["test_util.h"],
    deps = [
        "//tensorstore/internal:logging",
        "@com_google_protobuf//:protobuf",
    ],
)

tensorstore_cc_library(
    name = "protobuf_matchers",
    testonly = 1,
    srcs = ["protobuf_matchers.cc"],
    hdrs = ["protobuf_matchers.h"],
    deps = [
        "//tensorstore/internal:logging",
        "//tensorstore/util:assert_macros",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_protobuf//:protobuf",
        "@com_google_re2//:re2",
    ],
)
