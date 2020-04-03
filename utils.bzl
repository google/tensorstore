# Copyright 2020 The TensorStore Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@bazel_skylib//lib:selects.bzl", "selects")

def _sorted_unique_strings(x):
    return sorted({k: True for k in x}.keys())

def escape_target_name(name):
    name = name.replace("~", "~~")
    name = name.replace(":", "~c")
    name = name.replace("/", "~/")
    return name

def constraint_values_config_setting(constraint_values):
    constraint_values = _sorted_unique_strings(constraint_values)
    if len(constraint_values) == 0:
        return "//conditions:default"
    name = "constraints=" + ",".join([escape_target_name(x) for x in constraint_values])
    if native.existing_rule(name) == None:
        native.config_setting(
            name = name,
            constraint_values = constraint_values,
            visibility = ["//visibility:private"],
        )
    return name

# https://github.com/bazelbuild/proposals/blob/master/designs/2018-11-09-config-setting-chaining.md
def all_conditions(conditions):
    conditions = _sorted_unique_strings([x for x in conditions if x != "//conditions:default"])
    if len(conditions) == 0:
        return "//conditions:default"
    if len(conditions) == 1:
        return conditions[0]
    name = "all=" + ",".join([escape_target_name(x) for x in conditions])
    if native.existing_rule(name) == None:
        selects.config_setting_group(
            name = name,
            match_all = conditions,
        )
    return name

def repository_source_root():
    repo_name = native.repository_name().lstrip("@")
    if repo_name == "":
        return ""
    else:
        return "external/%s" % repo_name

def package_source_root():
    root = repository_source_root()
    p = native.package_name()
    if root != "" and p != "":
        root += "/"
    p = root + p
    return p

def package_relative_path(path):
    root = package_source_root()
    if path == ".":
        path = root
    else:
        path = root + "/" + path
    if path == "":
        path = "."
    return path

# Rule for simple expansion of template files. This performs a simple
# search over the template file for the keys in substitutions,
# and replaces them with the corresponding values.
#
# Typical usage:
#   load("/tools/build_rules/template_rule", "expand_header_template")
#   template_rule(
#       name = "ExpandMyTemplate",
#       src = "my.template",
#       out = "my.txt",
#       substitutions = {
#         "$VAR1": "foo",
#         "$VAR2": "bar",
#       }
#   )
#
# Args:
#   name: The name of the rule.
#   template: The template file to expand
#   out: The destination of the expanded file
#   substitutions: A dictionary mapping strings to their substitutions
#
# (Copied from tensorflow)
def template_rule_impl(ctx):
    ctx.actions.expand_template(
        template = ctx.file.src,
        output = ctx.outputs.out,
        substitutions = ctx.attr.substitutions,
    )

template_rule = rule(
    attrs = {
        "src": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
        "substitutions": attr.string_dict(mandatory = True),
        "out": attr.output(mandatory = True),
    },
    # output_to_genfiles is required for header files.
    output_to_genfiles = True,
    implementation = template_rule_impl,
)

# Workaround https://github.com/bazelbuild/bazel/issues/6337 by declaring
# the dependencies without strip_include_prefix.
def cc_library_with_strip_include_prefix(name, hdrs, deps = None, **kwargs):
    strip_include_prefix_name = name + "_strip_include_prefix_hack"
    if deps == None:
        deps = []
    deps = deps + [":" + strip_include_prefix_name]
    native.cc_library(
        name = strip_include_prefix_name,
        hdrs = hdrs,
        visibility = ["//visibility:private"],
    )
    native.cc_library(
        name = name,
        hdrs = hdrs,
        deps = deps,
        **kwargs
    )
