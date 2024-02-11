# Rules used by com_google_googleapis to build additional targets.
# See: https://github.com/googleapis/googleapis/blob/master/repository_rules.bzl

load(
    "@com_google_protobuf//bazel:upb_proto_library.bzl",
    _upb_proto_library = "upb_proto_library",
    _upb_proto_reflection_library = "upb_proto_reflection_library",
)
load(
    "@tensorstore//bazel:cc_grpc_library.bzl",
    _cc_grpc_library = "cc_grpc_library",
)
load(
    "@tensorstore//bazel:tensorstore.bzl",
    _tensorstore_cc_proto_library = "tensorstore_cc_proto_library",
)

#
# Common
#
def proto_library_with_info(**kwargs):
    pass

def moved_proto_library(**kwargs):
    pass

#
# Java
#
def java_proto_library(**kwargs):
    pass

def java_grpc_library(**kwargs):
    pass

def java_gapic_library(**kwargs):
    pass

def java_gapic_test(**kwargs):
    pass

def java_gapic_assembly_gradle_pkg(**kwargs):
    pass

#
# Python
#
def py_proto_library(**kwargs):
    pass

def py_grpc_library(**kwargs):
    pass

def py_gapic_library(**kwargs):
    pass

def py_test(**kwargs):
    pass

def py_gapic_assembly_pkg(**kwargs):
    pass

def py_import(**kwargs):
    pass

#
# Go
#
def go_proto_library(**kwargs):
    pass

def go_library(**kwargs):
    pass

def go_test(**kwargs):
    pass

def go_gapic_library(**kwargs):
    pass

def go_gapic_assembly_pkg(**kwargs):
    pass

#
# C++
#
def cc_proto_library(name, deps, **kwargs):
    _tensorstore_cc_proto_library(name = name, deps = deps)
    if name.endswith("_cc_proto"):
        name = name[:-9]

    # inject upb/upbdefs because CMake cannot use aspects to collect deps.
    _upb_proto_library(name = name + "_upb_proto", deps = deps)
    _upb_proto_reflection_library(name = name + "_upbdefs_proto", deps = deps)

cc_grpc_library = _cc_grpc_library

def cc_gapic_library(**kwargs):
    pass

#
# PHP
#
def php_proto_library(**kwargs):
    pass

def php_grpc_library(**kwargs):
    pass

def php_gapic_library(**kwargs):
    pass

def php_gapic_assembly_pkg(**kwargs):
    pass

#
# Node.js
#
def nodejs_gapic_library(**kwargs):
    pass

def nodejs_gapic_assembly_pkg(**kwargs):
    pass

#
# Ruby
#
def ruby_proto_library(**kwargs):
    pass

def ruby_grpc_library(**kwargs):
    pass

def ruby_ads_gapic_library(**kwargs):
    pass

def ruby_cloud_gapic_library(**kwargs):
    pass

def ruby_gapic_assembly_pkg(**kwargs):
    pass

#
# C#
#
def csharp_proto_library(**kwargs):
    pass

def csharp_grpc_library(**kwargs):
    pass

def csharp_gapic_library(**kwargs):
    pass

def csharp_gapic_assembly_pkg(**kwargs):
    pass
