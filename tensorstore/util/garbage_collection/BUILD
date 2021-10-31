load("//tensorstore:tensorstore.bzl", "tensorstore_cc_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_library(
    name = "garbage_collection",
    hdrs = [
        "fwd.h",
        "garbage_collection.h",
        "std_array.h",
        "std_map.h",
        "std_optional.h",
        "std_pair.h",
        "std_set.h",
        "std_tuple.h",
        "std_vector.h",
    ],
    deps = ["//tensorstore/util/apply_members"],
)

tensorstore_cc_library(
    name = "json",
    hdrs = ["json.h"],
    deps = [
        ":garbage_collection",
        "//tensorstore/internal:json_fwd",
    ],
)

tensorstore_cc_library(
    name = "protobuf",
    hdrs = ["protobuf.h"],
    deps = [
        ":garbage_collection",
        "@com_google_protobuf//:protobuf_lite",
    ],
)
