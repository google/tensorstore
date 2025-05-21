// Copyright 2025 The TensorStore Authors
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

#ifndef TENSORSTORE_KVSTORE_AUTO_DETECT_H_
#define TENSORSTORE_KVSTORE_AUTO_DETECT_H_

#include <stddef.h>

#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/strings/cord.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_kvstore {

// Specifies an auto-detection match result.
//
// Currently just specifies a URL scheme but may be extended with additional
// fields in the future.
struct AutoDetectMatch {
  /// Registered URL scheme of the format.
  std::string scheme;

  template <typename Sink>
  friend void AbslStringify(Sink&& sink, const AutoDetectMatch& match) {
    sink.Append(match.scheme);
  }

  friend bool operator==(const AutoDetectMatch& a, const AutoDetectMatch& b) {
    return a.scheme == b.scheme;
  }

  friend bool operator!=(const AutoDetectMatch& a, const AutoDetectMatch& b) {
    return !(a == b);
  }
};

// Parameters for `AutoDetectDirectoryMatcher`.
struct AutoDetectDirectoryOptions {
  // Set of relative filenames that are present.
  //
  // Includes any filenames in `AutoDetectDirectorySpec::filenames` that are
  // present.  May include additional filenames not in
  // `AutoDetectDirectorySpec::filenames`.
  absl::btree_set<std::string> filenames;
};

// Function for checking if a given directory format matches.
//
// Returns a list of matching formats.
using AutoDetectDirectoryMatcher = std::function<std::vector<AutoDetectMatch>(
    const AutoDetectDirectoryOptions& options)>;

// Specifies an auto-detector for a directory format.
struct AutoDetectDirectorySpec {
  // Set of relative filenames to check for existence.
  absl::btree_set<std::string> filenames;

  // Matcher function.
  AutoDetectDirectoryMatcher match;

  // Returns a simple detector that returns `{AutoDetectMatch{scheme}}` if
  // `filename` is present.
  static AutoDetectDirectorySpec SingleFile(std::string_view scheme,
                                            std::string_view filename);
};

// Parameters for `AutoDetectFileMatcher`.
struct AutoDetectFileOptions {
  // Prefix of the file.  Contains at least `AutoDetectFileSpec::prefix_length`
  // bytes, unless the file size is shorter in which case it contains the entire
  // file.

  absl::Cord prefix;
  // Suffix of the file.  Contains at least `AutoDetectFileSpec::prefix_length`
  // bytes, unless the file size is shorter in which case it contains the entire
  // file.  May be empty if the kvstore does not support suffix reads.
  absl::Cord suffix;
};

// Function for checking if a given directory format matches.
//
// Returns a list of matching formats.
using AutoDetectFileMatcher = std::function<std::vector<AutoDetectMatch>(
    const AutoDetectFileOptions& options)>;

// Specifies an auto-detector for a single-file format.
struct AutoDetectFileSpec {
  // Prefix length required for detection.
  size_t prefix_length = 0;

  // Suffix length required for detection.
  size_t suffix_length = 0;

  // Matcher function.
  AutoDetectFileMatcher match;

  // Returns a simple detector that returns `{AutoDetectMatch{scheme}}` if the
  // file starts with `signature`.
  static AutoDetectFileSpec PrefixSignature(std::string_view scheme,
                                            std::string_view signature);

  // Returns a simple detector that returns `{AutoDetectMatch{scheme}}` if the
  // first `signature_length` bytes of the file matches `predicate`.
  static AutoDetectFileSpec PrefixSignature(
      std::string_view scheme, size_t signature_length,
      std::function<bool(std::string_view signature)> predicate);

  // Returns a simple detector that returns `{AutoDetectMatch{scheme}}` if the
  // file ends with `signature`.
  static AutoDetectFileSpec SuffixSignature(std::string_view scheme,
                                            std::string_view signature);

  // Returns a simple detector that returns `{AutoDetectMatch{scheme}}` if the
  // last `signature_length` bytes of the file matches `predicate`.
  static AutoDetectFileSpec SuffixSignature(
      std::string_view scheme, size_t signature_length,
      std::function<bool(std::string_view signature)> predicate);
};

// Registers an auto-detector.  This should normally be used as a global
// variable.
struct AutoDetectRegistration {
  AutoDetectRegistration(AutoDetectFileSpec&& file_spec);
  AutoDetectRegistration(AutoDetectDirectorySpec&& directory_spec);

  // Clears all registrations, only intended for tests.
  static void ClearRegistrations();
};

// Auto-detects the format of the specified `base` kvstore.
//
// Uses `executor` for async continuations.
//
// If `base.path` looks like a file path (non-empty and doesn't end with "/"),
// detects either a registered file format, or a registered directory format
// once "/" is appended to `base.path`.
//
// If `base.path` is a directory path (empty or ends with "/"), detects only
// registered directory formats.
//
// If no formats are detected, returns an empty list of matches if no read
// errors occurred, or one of the read errors that ocurred.
//
// If at least one format is detected, any read errors (which may just be
// spurious errors due to files not being found) are ignored.
Future<std::vector<AutoDetectMatch>> AutoDetectFormat(Executor executor,
                                                      KvStore base);

}  // namespace internal_kvstore
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_AUTO_DETECT_H_
