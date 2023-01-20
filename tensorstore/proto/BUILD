load(
    "//bazel:tensorstore.bzl",
    "tensorstore_cc_library",
    "tensorstore_cc_proto_library",
    "tensorstore_cc_test",
    "tensorstore_proto_library",
)

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
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/status",
        "@com_google_riegeli//riegeli/bytes:string_reader",
        "@com_google_riegeli//riegeli/bytes:string_writer",
    ],
)

tensorstore_cc_test(
    name = "array_test",
    size = "small",
    srcs = ["array_test.cc"],
    deps = [
        ":array",
        ":array_cc_proto",
        ":protobuf_matchers",
        "//tensorstore:array",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:strided_layout",
        "//tensorstore/index_space:index_transform_testutil",
        "//tensorstore/internal:data_type_random_generator",
        "//tensorstore/internal:test_util",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "encode_time",
    srcs = ["encode_time.cc"],
    hdrs = ["encode_time.h"],
    deps = [
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/time",
        "@com_google_protobuf//:protobuf",
    ],
)

tensorstore_cc_test(
    name = "encode_time_test",
    srcs = ["encode_time_test.cc"],
    deps = [
        ":encode_time",
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
        "//tensorstore/internal/json:array",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:dimension_indexed",
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_test(
    name = "index_transform_test",
    size = "small",
    srcs = ["index_transform_test.cc"],
    deps = [
        ":index_transform",
        ":index_transform_cc_proto",
        ":protobuf_matchers",
        "//tensorstore:array",
        "//tensorstore:box",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore:rank",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/status",
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
    size = "small",
    srcs = ["schema_test.cc"],
    deps = [
        ":protobuf_matchers",
        ":schema",
        ":schema_cc_proto",
        "//tensorstore:schema",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "protobuf_matchers",
    testonly = 1,
    srcs = ["protobuf_matchers.cc"],
    hdrs = ["protobuf_matchers.h"],
    deps = [
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_protobuf//:protobuf",
        "@com_google_re2//:re2",
    ],
)

tensorstore_cc_library(
    name = "proto_util",
    srcs = ["proto_util.cc"],
    hdrs = ["proto_util.h"],
    deps = [
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
    ],
)

tensorstore_cc_test(
    name = "proto_util_test",
    size = "small",
    srcs = ["proto_util_test.cc"],
    deps = [
        ":array_cc_proto",
        ":proto_util",
        ":protobuf_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "proto_binder",
    srcs = ["proto_binder.cc"],
    hdrs = ["proto_binder.h"],
    deps = [
        ":proto_util",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/internal:json_fwd",
        "//tensorstore/internal:type_traits",
        "//tensorstore/internal/json:value_as",
        "//tensorstore/internal/json_binding",
        "//tensorstore/util:str_cat",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/status",
        "@com_google_protobuf//:protobuf",
    ],
)

tensorstore_cc_test(
    name = "proto_binder_test",
    size = "small",
    srcs = ["proto_binder_test.cc"],
    deps = [
        ":array_cc_proto",
        ":proto_binder",
        ":protobuf_matchers",
        "//tensorstore:json_serialization_options",
        "//tensorstore/internal:json_fwd",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "parse_text_proto_or_die",
    hdrs = ["parse_text_proto_or_die.h"],
    deps = [
        ":proto_util",
        "//tensorstore/internal:source_location",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
    ],
)
