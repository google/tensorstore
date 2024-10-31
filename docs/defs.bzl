load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "CPP_COMPILE_ACTION_NAME")
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")

# Aspect for extracting the C++ compiler flags that apply to a target.
CompilationAspect = provider()

def _compilation_flags_aspect_impl(target, ctx):
    cc_toolchain = find_cpp_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    compile_variables = cc_common.create_compile_variables(
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        user_compile_flags = ctx.fragments.cpp.cxxopts +
                             ctx.fragments.cpp.copts,
        add_legacy_cxx_options = True,
    )
    compiler_options = cc_common.get_memory_inefficient_command_line(
        feature_configuration = feature_configuration,
        action_name = CPP_COMPILE_ACTION_NAME,
        variables = compile_variables,
    )
    return [CompilationAspect(compiler_options = compiler_options)]

compilation_flags_aspect = aspect(
    attrs = {
        "_cc_toolchain": attr.label(
            default = Label("@bazel_tools//tools/cpp:current_cc_toolchain"),
        ),
        "_xcode_config": attr.label(default = Label("@bazel_tools//tools/osx:current_xcode_config")),
    },
    fragments = ["cpp"],
    provides = [CompilationAspect],
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
    implementation = _compilation_flags_aspect_impl,
)

# Rule for transitioning platform in order to switch to a hermetic clang toolchain.
def _compiler_transition_impl(settings, attr):
    cpp_compiler_constraint = attr.cpp_compiler_constraint
    if cpp_compiler_constraint == None:
        return {}
    return {"//command_line_option:platforms": str(cpp_compiler_constraint)}

compiler_transition = transition(
    implementation = _compiler_transition_impl,
    inputs = [],
    outputs = ["//command_line_option:platforms"],
)

def _cc_preprocessed_output_impl(ctx):
    target = ctx.attr.target

    # When a transition is specified for an attribute, the attribute always
    # becomes a list.
    # https://bazel.build/extending/config#accessing-attributes-with-transitions
    if type(target) == type([]):
        target = target[0]

    compilation_outputs = target.output_groups.compilation_outputs.to_list()
    if len(compilation_outputs) != 1:
        fail("More than one compilation output: ", compilation_outputs)
    out = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.symlink(output = out, target_file = compilation_outputs[0])
    ctx.actions.write(output = ctx.outputs.flags_output, content = json.encode(target[CompilationAspect].compiler_options))
    return [DefaultInfo(
        files = depset([out]),
        runfiles = ctx.runfiles(files = [out]),
    )]

# Collects the preprocessed output of a C++ source file and the compilation
# flags in JSON format.
#
# The `target` attribute must refer to a `cc_library` label that specifies
# `copts=["-E"]` (among others) to ensure preprocssed output rather than object
# file output.
#
# The default output of the new target defined by this rule will be a symlink to
# the preprocessed output of `target`. The additional output file named by the
# `flags_output` attribute will be a JSON file containing a single array of
# strings specifying the compilation flags that would be used to build `target`.
cc_preprocessed_output = rule(
    implementation = _cc_preprocessed_output_impl,
    attrs = {
        "target": attr.label(
            aspects = [compilation_flags_aspect],
            cfg = compiler_transition,
        ),
        "flags_output": attr.output(doc = "Name of the json file to which the compiler flags are written."),
        "cpp_compiler_constraint": attr.label(),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
    provides = [DefaultInfo],
)
