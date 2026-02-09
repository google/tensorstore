workspace(
    name = "tensorstore",
)

load("//:external.bzl", "tensorstore_dependencies")

tensorstore_dependencies()

load("@bazel_features//:deps.bzl", "bazel_features_deps")

bazel_features_deps()

load("@rules_cc//cc:extensions.bzl", "compatibility_proxy_repo")

compatibility_proxy_repo()

load("@rules_shell//shell:repositories.bzl", "rules_shell_toolchains")

rules_shell_toolchains()

# @rules_python configuration for local python runtime / toolchain.
#
# py_repositories() creates internal repositories for python, and use of rules_python
# fails when they are not present.
load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

# Register the local toolchains for python:
#
# Step 1: Define the python runtime.
#  This is done by local_python_runtime() in tensorstore_dependencies()
#
# Step 2: Create toolchains for the runtimes
#  This is done by local_runtime_toolchains_repo()
#
# Step 3: Register the toolchains
#  This is done by register_toolchains()
load("@rules_python//python/local_toolchains:repos.bzl", "local_runtime_toolchains_repo")

local_runtime_toolchains_repo(
    name = "local_toolchains",
    runtimes = ["local_config_python"],
)

register_toolchains("@local_toolchains//:all")

# Register proto toolchains.
load("@rules_proto//proto:toolchains.bzl", "rules_proto_toolchains")

rules_proto_toolchains()

# Register build_bazel_apple_support toolchains, which are needed for cross-compilaton
# macOS. Unfortunately this (small) repo will have to be downloaded in all
# cases, even though it is only needed on macOS when cross-compiling.
load(
    "@build_bazel_apple_support//lib:repositories.bzl",
    "apple_support_dependencies",
)

apple_support_dependencies()

# Define LLVM toolchain used for extracting C++ API documentation information
load("@toolchains_llvm//toolchain:rules.bzl", "llvm_toolchain")

llvm_toolchain(
    name = "llvm_toolchain",
    extra_target_compatible_with = {
        "": ["@//docs:docs_toolchain_value"],
    },
    # https://github.com/bazel-contrib/toolchains_llvm/blob/master/toolchain/internal/llvm_distributions.bzl
    llvm_versions = {
        # Note: Older versions are built against older glibc, which is needed
        # for compatibility with manylinux containers.
        "": "15.0.6",
        "darwin-aarch64": "15.0.7",
        "darwin-x86_64": "15.0.7",
    },
)

load("@llvm_toolchain//:toolchains.bzl", "llvm_register_toolchains")

llvm_register_toolchains()

# Register the default @rules_nasm toolchain.
register_toolchains("@rules_nasm//nasm/toolchain")
