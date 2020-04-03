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

#include "absl/strings/string_view.h"

namespace tensorstore {
namespace internal_path {
// Implementation of JoinPath
std::string JoinPathImpl(std::initializer_list<absl::string_view> paths);
}  // namespace internal_path

namespace internal {
// Join multiple paths together, without introducing unnecessary path
// separators.
// For example:
//
//  Arguments                  | JoinPath
//  ---------------------------+----------
//  '/foo', 'bar'              | /foo/bar
//  '/foo/', 'bar'             | /foo/bar
//  '/foo', '/bar'             | /foo/bar
//  '/foo/', '/bar'            | /foo/bar

//
// Usage:
//   std::string path = JoinPath("/mydir", filename);
//   std::string path = JoinPath(FLAGS_test_srcdir, filename);
//   std::string path = JoinPath("/full", "path", "to", "filename);
template <typename... T>
std::string JoinPath(const T&... args) {
  return internal_path::JoinPathImpl({args...});
}

// Create a URI from a `scheme`, `host`, and `path`. If the scheme
// is empty, then the result is the path.
std::string CreateURI(absl::string_view scheme, absl::string_view host,
                      absl::string_view path);

// Parse a `uri`, populating the `scheme`, `host`, and `path`.
// If the `uri` is invalid, then the scheme and host are set as empty strings,
// and the entire `uri` is returned as a path.
//
// If the `uri` has no path, then the scheme and host are set, but no path.
void ParseURI(absl::string_view uri, absl::string_view* scheme,
              absl::string_view* host, absl::string_view* path);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_PATH_H_
