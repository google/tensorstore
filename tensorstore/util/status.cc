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

#if !defined(TENSORSTORE_INTERNAL_STATUS_TEST_HACK)
// Facilitate an internal test by not including status.h
#include "tensorstore/util/status.h"
#endif

#include <cstdio>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorstore/internal/source_location.h"

namespace tensorstore {
namespace internal {

absl::Status MaybeAnnotateStatusImpl(absl::Status source,
                                     std::string_view message,
                                     std::optional<SourceLocation> loc) {
  if (source.ok()) return source;

  absl::Status dest(
      source.code(),  //
      source.message().empty()
          ? message
          : std::string_view(absl::StrCat(message, ": ", source.message())));

  // Preserve the payloads.
  source.ForEachPayload([&](absl::string_view name, const absl::Cord& value) {
    dest.SetPayload(name, value);
  });

  /// TODO: Consider adding the source locations to the status,
  /// however if we do, it may cause some tests to fail by changing the
  /// status.ToString() output.
  //  dest.SetPayload(
  //      absl::StrFormat("loc/%02d", count + 1),
  //      absl::Cord(absl::StrFormat("%s:%d", loc.file_name(), loc.line())));
  return dest;
}

[[noreturn]] void FatalStatus(const char* message, const absl::Status& status,
                              SourceLocation loc) {
  std::fprintf(stderr, "%s:%d: %s: %s\n", loc.file_name(), loc.line(), message,
               status.ToString().c_str());
  std::terminate();
}

}  // namespace internal
}  // namespace tensorstore
