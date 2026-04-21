load(
    "@tensorstore//bazel:tensorstore.bzl",
    _tensorstore_cc_proto_library = "tensorstore_cc_proto_library",
)

def pgv_go_proto_library(**kwargs):
    pass

def pgv_cc_proto_library(
        name,
        deps = [],
        **kargs):
    # Typically invoked via https://github.com/envoyproxy/data-plane-api/blob/main/bazel/api_build_system.bzl
    _tensorstore_cc_proto_library(
        name = name,
        deps = deps,
    )

def pgv_java_proto_library(**kwargs):
    pass
