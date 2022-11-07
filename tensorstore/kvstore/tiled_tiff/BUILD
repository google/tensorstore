# libtiff-backed KeyValueStore driver

load("//tensorstore:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

filegroup(
    name = "doc_sources",
    srcs = glob([
        "**/*.rst",
        "**/*.yml",
    ]),
)

tensorstore_cc_library(
    name = "tiled_tiff",
    srcs = ["tiled_tiff_key_value_store.cc"],
    hdrs = ["omexml.h"],
    deps = [
        ":omexml",
        "//tensorstore/kvstore/file:file_util",
        "//tensorstore/kvstore/file:util",
        "//tensorstore:context",
        "//tensorstore/internal:context_binding",
        "//tensorstore/internal:file_io_concurrency_resource",
        "//tensorstore/internal:flat_cord_builder",
        "//tensorstore/internal:os_error_code",
        "//tensorstore/internal:path",
        "//tensorstore/internal:type_traits",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/internal/metrics",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:any_receiver",
        "//tensorstore/util/execution:sender",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/time",
        "@libtiff//:tiff",
    ],
    alwayslink = 1,
)

tensorstore_cc_library(
    name = "omexml",
    srcs = ["omexml.cc"],
    hdrs = ["omexml.h"],
    deps = ["@com_github_pugixml//:pugixml",]
)
