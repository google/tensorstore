load("//bazel:tensorstore.bzl", "tensorstore_cc_binary", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_test(
    name = "compute_percentiles",
    size = "small",
    srcs = [
        "compute_percentiles.cc",
        "data_type_invoker.h",
    ],
    tags = ["manual"],
    deps = [
        "//tensorstore",
        "//tensorstore:all_drivers",
        "//tensorstore:array",
        "//tensorstore:context",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:open",
        "//tensorstore:open_mode",
        "//tensorstore:progress",
        "//tensorstore:rank",
        "//tensorstore:spec",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:transformed_array",
        "//tensorstore/internal:init_tensorstore",
        "//tensorstore/util:future",
        "//tensorstore/util:iterate_over_index_range",
        "//tensorstore/util:json_absl_flag",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "//tensorstore/util:utf8_string",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:marshalling",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@net_sourceforge_half//:half",
    ],
)

tensorstore_cc_binary(
    name = "create_array",
    srcs = ["create_array.cc"],
    deps = [
        "//tensorstore:array",
        "//tensorstore:index",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:transformed_array",
        "//tensorstore/util:iterate_over_index_range",
        "//tensorstore/util:status",
    ],
)

tensorstore_cc_binary(
    name = "image_convolution",
    srcs = ["image_convolution.cc"],
    deps = [
        "//tensorstore:array",
        "//tensorstore:index",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:transformed_array",
        "//tensorstore/util:iterate_over_index_range",
        "//tensorstore/util:status",
        "@com_google_absl//absl/functional:function_ref",
    ],
)

tensorstore_cc_binary(
    name = "map_apply",
    srcs = ["map_apply.cc"],
    deps = [
        "//tensorstore:array",
        "//tensorstore:index",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:transformed_array",
        "//tensorstore/util:iterate_over_index_range",
        "//tensorstore/util:status",
    ],
)

tensorstore_cc_binary(
    name = "status_result",
    srcs = ["status_result.cc"],
    deps = [
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_binary(
    name = "extract_slice",
    srcs = ["extract_slice.cc"],
    deps = [
        "//tensorstore",
        "//tensorstore:all_drivers",
        "//tensorstore:array",
        "//tensorstore:context",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:open",
        "//tensorstore:open_mode",
        "//tensorstore:spec",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:init_tensorstore",
        "//tensorstore/internal/image",
        "//tensorstore/internal/image:avif",
        "//tensorstore/internal/image:jpeg",
        "//tensorstore/internal/image:png",
        "//tensorstore/internal/image:webp",
        "//tensorstore/util:json_absl_flag",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_riegeli//riegeli/bytes:fd_writer",
        "@com_google_riegeli//riegeli/bytes:std_io",
    ],
)
