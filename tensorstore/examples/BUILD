load("//tensorstore:tensorstore.bzl", "tensorstore_cc_binary")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

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
    name = "create_array",
    srcs = ["create_array.cc"],
    deps = [
        "//tensorstore:array",
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
    ],
)
