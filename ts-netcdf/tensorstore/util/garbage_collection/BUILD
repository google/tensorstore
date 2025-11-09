load("//bazel:tensorstore.bzl", "tensorstore_cc_library")

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
    deps = [
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/util:result",
        "//tensorstore/util/apply_members",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/time",
    ],
)

tensorstore_cc_library(
    name = "json",
    hdrs = ["json.h"],
    deps = [
        ":garbage_collection",
        "@nlohmann_json//:json",
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
