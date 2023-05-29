// Copyright 2020 The TensorStore Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORSTORE_INTERNAL_PATH_H_
#define TENSORSTORE_INTERNAL_PATH_H_

#include <initializer_list>
#include <string>
#include <string_view>

namespace tensorstore {
namespace internal_path {
// Implementation of JoinPath
std::string JoinPathImpl(std::initializer_list<std::string_view> paths);
}  // namespace internal_path

namespace internal {
// Join multiple paths together, without introducing unnecessary path
// separators.
//
// Equivalent to repeatedly calling `AppendPathComponent` for each argument,
// starting with an empty string.
//
// For example:
//
//  Arguments                  | JoinPath
//  ---------------------------+----------
//  '/foo', 'bar'              | /foo/bar
//  '/foo/', 'bar'             | /foo/bar
//  '/foo', '/bar'             | /foo/bar
//  '/foo/', '/bar'            | /foo//bar

//
// Usage:
//   std::string path = JoinPath("/mydir", filename);
//   std::string path = JoinPath(FLAGS_test_srcdir, filename);
//   std::string path = JoinPath("/full", "path", "to", "filename);
template <typename... T>
std::string JoinPath(const T&... args) {
  return internal_path::JoinPathImpl({args...});
}

// Splits a path into the pair {dirname, basename} using platform-specific
// path separator ([/\] on Windows, otherwise [/]).
std::pair<std::string_view, std::string_view> PathDirnameBasename(
    std::string_view path);

/// Joins `component` to the end of `path`, adding a '/'-separator between them
/// if both are non-empty and `path` does not in end '/' and `component` does
/// not start with '/'.
void AppendPathComponent(std::string& path, std::string_view component);

/// Ensure that `path` is either empty, or consists of a non-empty string
/// followed by a '/'.
void EnsureDirectoryPath(std::string& path);

/// Ensure that `path` does not end in '/'.
void EnsureNonDirectoryPath(std::string& path);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_PATH_H_
