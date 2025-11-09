# Copyright 2025 The TensorStore Authors
#
# Licensed under the Apache License, Version 2.0 (self,the "License");
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

from .provider import Provider

# FeatureConfiguration
# CcToolchainConfigInfo
# CompilationContext
# CcCompilationOutputs
# Variables
# LibraryToLink
# Variables
# LinkerInput
# LinkingContext
# LtoCompilationContext
# CcLinkingOutputs
# CompilationContext
# CcCompilationOutputs


class CcToolchainInfo(Provider):

  pass


class BazelModuleCcCommon:

  do_not_use_tools_cpp_compiler_present = True

  CcToolchainInfo = staticmethod(CcToolchainInfo)  # type: ignore[not-callable]

  def action_is_enabled(self, feature_configuration, action_name):
    pass

  def compile_fork(
      self,
      actions,
      feature_configuration,
      cc_toolchain,
      srcs=None,
      public_hdrs=None,
      private_hdrs=None,
      includes=None,
      quote_includes=None,
      system_includes=None,
      framework_includes=None,
      defines=None,
      local_defines=None,
      include_prefix='',
      strip_include_prefix='',
      user_compile_flags=None,
      conly_flags=None,
      cxx_flags=None,
      compilation_contexts=None,
      name=None,
      disallow_pic_outputs=False,
      disallow_nopic_outputs=False,
      additional_inputs=None,
      module_interfaces=None,
  ):
    pass

  def compile(
      self,
      actions,
      feature_configuration,
      cc_toolchain,
      srcs=None,
      public_hdrs=None,
      private_hdrs=None,
      includes=None,
      quote_includes=None,
      system_includes=None,
      framework_includes=None,
      defines=None,
      local_defines=None,
      include_prefix='',
      strip_include_prefix='',
      user_compile_flags=None,
      conly_flags=None,
      cxx_flags=None,
      compilation_contexts=None,
      name=None,
      disallow_pic_outputs=False,
      disallow_nopic_outputs=False,
      additional_inputs=None,
      module_interfaces=None,
  ):
    pass

  def configure_features(
      self,
      ctx=None,
      cc_toolchain=None,
      language=None,
      requested_features=None,
      unsupported_features=None,
  ):
    pass

  def create_cc_toolchain_config_info(
      self,
      ctx,
      features=None,
      action_configs=None,
      artifact_name_patterns=None,
      cxx_builtin_include_directories=None,
      toolchain_identifier=None,
      host_system_name=None,
      target_system_name=None,
      target_cpu=None,
      target_libc=None,
      compiler=None,
      abi_version=None,
      abi_libc_version=None,
      tool_paths=None,
      make_variables=None,
      builtin_sysroot=None,
  ):
    pass

  def create_compilation_context(
      self,
      headers=None,
      system_includes=None,
      includes=None,
      quote_includes=None,
      framework_includes=None,
      defines=None,
      local_defines=None,
  ):
    pass

  def create_compilation_outputs(self, objects=None, pic_objects=None):
    pass

  def create_compile_variables(
      self,
      cc_toolchain,
      feature_configuration,
      source_file=None,
      output_file=None,
      user_compile_flags=None,
      include_directories=None,
      quote_include_directories=None,
      system_include_directories=None,
      framework_include_directories=None,
      preprocessor_defines=None,
      thinlto_index=None,
      thinlto_input_bitcode_file=None,
      thinlto_output_object_file=None,
      use_pic=False,
      add_legacy_cxx_options=False,
      variables_extension=None,
  ):
    pass

  def create_library_to_link(
      self,
      actions,
      feature_configuration=None,
      cc_toolchain=None,
      static_library=None,
      pic_static_library=None,
      dynamic_library=None,
      interface_library=None,
      pic_objects=None,
      objects=None,
      alwayslink=False,
      dynamic_library_symlink_path='',
      interface_library_symlink_path='',
  ):
    pass

  def create_link_variables(
      self,
      cc_toolchain,
      feature_configuration,
      library_search_directories=None,
      runtime_library_search_directories=None,
      user_link_flags=None,
      output_file=None,
      param_file=None,
      is_using_linker=True,
      is_linking_dynamic_library=False,
      must_keep_debug=True,
      use_test_only_flags=False,
      is_static_linking_mode=True,
  ):
    pass

  def create_linker_input(
      self, owner, libraries=None, user_link_flags=None, additional_inputs=None
  ):
    pass

  def create_linking_context_from_compilation_outputs(
      self,
      actions,
      name,
      feature_configuration,
      cc_toolchain,
      language='c++',
      disallow_static_libraries=False,
      disallow_dynamic_library=False,
      compilation_outputs=None,
      linking_contexts=None,
      user_link_flags=None,
      alwayslink=False,
      additional_inputs=None,
      variables_extension=None,
  ):
    pass

  def create_linking_context(
      self,
      linker_inputs=None,
      libraries_to_link=None,
      user_link_flags=None,
      additional_inputs=None,
  ):
    pass

  def create_lto_compilation_context(self, objects=None):
    pass

  def get_environment_variables(
      self, feature_configuration, action_name, variables
  ):
    pass

  def get_execution_requirements(self, feature_configuration, action_name):
    pass

  def get_memory_inefficient_command_line(
      self, feature_configuration, action_name, variables
  ):
    pass

  def get_tool_for_action(self, feature_configuration, action_name):
    pass

  def is_enabled(self, feature_configuration, feature_name):
    pass

  def link(
      self,
      actions,
      name,
      feature_configuration,
      cc_toolchain,
      language='c++',
      output_type='executable',
      link_deps_statically=True,
      compilation_outputs=None,
      linking_contexts=None,
      user_link_flags=None,
      stamp=0,
      additional_inputs=None,
      additional_outputs=None,
      variables_extension=None,
  ):
    pass

  def merge_compilation_contexts(self, compilation_contexts=None):
    pass

  def merge_compilation_outputs(self, compilation_outputs=None):
    pass
