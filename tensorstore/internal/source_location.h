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

#ifndef TENSORSTORE_INTERNAL_SOURCE_LOCATION_H_
#define TENSORSTORE_INTERNAL_SOURCE_LOCATION_H_

#include <cstdint>
#include <utility>

#include "absl/base/config.h"

namespace tensorstore {

// TENSORSTORE_HAVE_SOURCE_LOCATION_CURRENT
//
// Indicates whether `SourceLocation::current()` will return useful
// information in some contexts.
#ifndef TENSORSTORE_HAVE_SOURCE_LOCATION_CURRENT
#if ABSL_HAVE_BUILTIN(__builtin_LINE) && ABSL_HAVE_BUILTIN(__builtin_FILE)
#define TENSORSTORE_HAVE_SOURCE_LOCATION_CURRENT 1
#elif defined(__GNUC__) && __GNUC__ >= 5
#define TENSORSTORE_HAVE_SOURCE_LOCATION_CURRENT 1
#elif defined(_MSC_VER) && _MSC_VER >= 1926
#define TENSORSTORE_HAVE_SOURCE_LOCATION_CURRENT 1
#else
#define TENSORSTORE_HAVE_SOURCE_LOCATION_CURRENT 0
#endif
#endif

/// Macros to correctly construct a ::tensorstore::SourceLocation object
/// with the current file and line.
///
/// When calling a method that takes a SourceLocation, use TENSORSTORE_LOC,
/// like:
///
///   ::tensorstore::internal::LogMessageFatal("Whoops!", TENSORSTORE_LOC).
///
/// When writing a function which *may* accept a SourceLocation, and by default
/// the call site should be used, write:
///
///  void TakesLoc(SourceLocation loc TENSORSTORE_LOC_CURRENT_DEFAULT_ARG);
///
/// NOTE: On some compiler versions, use of the method will require the caller
/// to explicitly pass TENSORSTORE_LOC
///
#if TENSORSTORE_HAVE_SOURCE_LOCATION_CURRENT
#define TENSORSTORE_LOC (::tensorstore::SourceLocation::current())
#define TENSORSTORE_LOC_CURRENT_DEFAULT_ARG \
  = ::tensorstore::SourceLocation::current()
#else
#define TENSORSTORE_LOC (::tensorstore::SourceLocation(__LINE__, __FILE__))
#define TENSORSTORE_LOC_CURRENT_DEFAULT_ARG
#endif

class SourceLocation {
 public:
  /// Do not call the constructor directly, instead use the macro
  /// TENSORSTORE_LOC to construct a SourceLocation.
  constexpr SourceLocation(std::uint_least32_t line, const char* file_name)
      : line_(line), file_name_(file_name) {}

#if TENSORSTORE_HAVE_SOURCE_LOCATION_CURRENT
  static constexpr SourceLocation current(
      std::uint_least32_t line = __builtin_LINE(),
      const char* file_name = __builtin_FILE()) {
    return SourceLocation(line, file_name);
  }
#else
  static constexpr SourceLocation current() {
    return SourceLocation(1, "<source_location>");
  }
#endif

  const char* file_name() const { return file_name_; }
  constexpr std::uint_least32_t line() const { return line_; }

 private:
  std::uint_least32_t line_;
  const char* file_name_;
};

}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_SOURCE_LOCATION_H_
