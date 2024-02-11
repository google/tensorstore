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

class SourceLocation {
  struct PrivateTag {
   private:
    explicit PrivateTag() = default;
    friend class SourceLocation;
  };

 public:
  // Avoid this constructor; it populates the object with filler values.
  // Instead, use `SourceLocation::current()` to construct SourceLocation.
  constexpr SourceLocation() : line_(1), file_name_("") {}

#if TENSORSTORE_HAVE_SOURCE_LOCATION_CURRENT
  // SourceLocation::current
  //
  // Creates a `SourceLocation` based on the current line and file.  APIs that
  // accept a `SourceLocation` as a default parameter can use this to capture
  // their caller's locations.
  //
  // Example:
  //
  //   void TracedAdd(int i, SourceLocation loc = SourceLocation::current()) {
  //     std::cout << loc.file_name() << ":" << loc.line() << " added " << i;
  //     ...
  //   }
  static constexpr SourceLocation current(
      PrivateTag = PrivateTag{}, std::uint_least32_t line = __builtin_LINE(),
      const char* file_name = __builtin_FILE()) {
    return SourceLocation(line, file_name);
  }
#else
  // Creates a fake `SourceLocation` of "<source_location>" at line number 1,
  // if no `SourceLocation::current()` implementation is available.
  static constexpr SourceLocation current() {
    return SourceLocation(1, "<source_location>");
  }
#endif

  const char* file_name() const { return file_name_; }
  constexpr std::uint_least32_t line() const { return line_; }

 private:
  constexpr SourceLocation(std::uint_least32_t line, const char* file_name)
      : line_(line), file_name_(file_name) {}

  std::uint_least32_t line_;
  const char* file_name_;
};

}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_SOURCE_LOCATION_H_
