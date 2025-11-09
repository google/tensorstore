# Modified version of:
# https://github.com/facebook/folly/blob/183e1994fe00af1230371f48e3e1fd7372c98d0f/CMake/FindZstd.cmake
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# - Try to find Facebook zstd library
# This will define
# ZSTD_FOUND
# ZSTD_INCLUDE_DIR
# ZSTD_LIBRARY
#

find_path(Zstd_INCLUDE_DIR NAMES zstd.h)

find_library(Zstd_LIBRARY_DEBUG NAMES zstdd zstd_staticd)
find_library(Zstd_LIBRARY_RELEASE NAMES zstd zstd_static)

include(SelectLibraryConfigurations)
SELECT_LIBRARY_CONFIGURATIONS(Zstd)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(
  Zstd DEFAULT_MSG
  Zstd_LIBRARY Zstd_INCLUDE_DIR
  )

set(ZSTD_FOUND ${Zstd_FOUND})

if (Zstd_FOUND)
  set(ZSTD_LIBRARY ${Zstd_LIBRARY})
  set(ZSTD_INCLUDE_DIR ${Zstd_INCLUDE_DIR})
  if(NOT TARGET Zstd::Zstd)
    add_library(Zstd::Zstd UNKNOWN IMPORTED)
    set_target_properties(Zstd::Zstd PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES ${Zstd_INCLUDE_DIR}
      IMPORTED_LINK_INTERFACE_LANGUAGES C)

    if(ZSTD_LIBRARY_RELEASE)
      set_property(TARGET Zstd::Zstd APPEND PROPERTY
        IMPORTED_CONFIGURATIONS RELEASE)
      set_target_properties(Zstd::Zstd PROPERTIES
        IMPORTED_LOCATION_RELEASE "${Zstd_LIBRARY_RELEASE}")
    endif()

    if(ZSTD_LIBRARY_DEBUG)
      set_property(TARGET Zstd::Zstd APPEND PROPERTY
        IMPORTED_CONFIGURATIONS DEBUG)
      set_target_properties(Zstd::Zstd PROPERTIES
        IMPORTED_LOCATION_DEBUG "${Zstd_LIBRARY_DEBUG}")
    endif()

    if(NOT ZSTD_LIBRARY_RELEASE AND NOT ZSTD_LIBRARY_DEBUG)
      set_target_properties(Zstd::Zstd PROPERTIES
        IMPORTED_LOCATION "${Zstd_LIBRARY}")
    endif()
  endif()
endif()

mark_as_advanced(Zstd_INCLUDE_DIR Zstd_LIBRARY ZSTD_INCLUDE_DIR ZSTD_LIBRARY)
