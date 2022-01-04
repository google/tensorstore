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

struct ParsedGenericUri {
  /// Portion of URI before the initial "://", or empty if there is no "://".
  std::string_view scheme;

  /// Portion of URI after the initial "://" (or from the beginning if there is
  /// no "://") and before the first `?` or `#`.  Not percent decoded.
  std::string_view authority_and_path;

  /// Portion of URI after the first `?` but before the first `#`.  Not percent
  /// decoded.
  std::string_view query;

  /// Portion of URI after the first `#`.  Not percent decoded.
  std::string_view fragment;
};

/// Parses a "generic" URI of the form
/// `<scheme>://<authority-and-path>?<query>#<fragment>` where the `?<query>`
/// and `#<fragment>` portions are optional.
ParsedGenericUri ParseGenericUri(std::string_view uri);

/// Joins `component` to the end of `path`, adding a '/'-separator between them
/// if both are non-empty and `path` does not in end '/' and `component` does
/// not start with '/'.
void AppendPathComponent(std::string& path, std::string_view component);

/// Ensure that `path` is either empty, or consists of a non-empty string
/// followed by a '/'.
void EnsureDirectoryPath(std::string& path);

/// Ensure that `path` does not end in '/'.
void EnsureNonDirectoryPath(std::string& path);

/// Decodes "%XY" sequences in `src`, where `X` and `Y` are hex digits, to the
/// corresponding character `\xXY`.  "%" characters not followed by 2 hex digits
/// are left unchanged.
///
/// Assigns the decoded result to `dest`.
void PercentDecode(std::string_view src, std::string& dest);

/// Same as above but returns the decoded result.
std::string PercentDecode(std::string_view src);

/// Percent-encodes characters not allowed in the URI path component, as defined
/// by RFC2396:
///
/// https://datatracker.ietf.org/doc/html/rfc2396
///
/// Allowed characters are:
///
/// - Unreserved characters: `unreserved` as defined by RFC2396
///   https://datatracker.ietf.org/doc/html/rfc2396#section-2.3
///   a-z, A-Z, 0-9, "-", "_", ".", "!", "~", "*", "'", "(", ")"
///
/// - Path characters: `pchar` as defined by RFC2396
///   https://datatracker.ietf.org/doc/html/rfc2396#section-3.3
///   ":", "@", "&", "=", "+", "$", ","
///
/// - Path segment parameter separator:
///   https://datatracker.ietf.org/doc/html/rfc2396#section-3.3
///   ";"
///
/// - Path segment separator:
///   https://datatracker.ietf.org/doc/html/rfc2396#section-3.3
///   "/"
void PercentEncodeUriPath(std::string_view src, std::string& dest);

/// Same as above but returns the encoded result.
std::string PercentEncodeUriPath(std::string_view src);

/// Percent-encodes characters not in the unreserved set, as defined by RFC2396:
///
/// Allowed characters are:
///
/// - Unreserved characters: `unreserved` as defined by RFC2396
///   https://datatracker.ietf.org/doc/html/rfc2396#section-2.3
///   a-z, A-Z, 0-9, "-", "_", ".", "!", "~", "*", "'", "(", ")"
///
/// This is equivalent to the ECMAScript `encodeURIComponent` function:
/// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/encodeURIComponent
void PercentEncodeUriComponent(std::string_view src, std::string& dest);

/// Same as above but returns the encoded result.
std::string PercentEncodeUriComponent(std::string_view src);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_PATH_H_
