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

#include <array>
#include <cstdio>
#include <exception>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorstore/internal/source_location.h"

namespace tensorstore {
namespace internal {

/// Add a source location to the status.
void MaybeAddSourceLocationImpl(absl::Status& status, SourceLocation loc) {
  constexpr const char kSourceLocationKey[] = "source locations";
#if TENSORSTORE_HAVE_SOURCE_LOCATION_CURRENT
  if (loc.line() <= 1) return;
  std::string_view filename(loc.file_name());
  if (auto idx = filename.find("tensorstore"); idx != std::string::npos) {
    filename.remove_prefix(idx);
  }

  auto payload = status.GetPayload(kSourceLocationKey);
  if (!payload.has_value()) {
    status.SetPayload(kSourceLocationKey, absl::Cord(absl::StrFormat(
                                              "%s:%d", filename, loc.line())));
  } else {
    payload->Append(absl::StrFormat("\n%s:%d", filename, loc.line()));
    status.SetPayload(kSourceLocationKey, std::move(*payload));
  }
#endif
}

absl::Status MaybeAnnotateStatusImpl(absl::Status source,
                                     std::string_view prefix_message,
                                     std::optional<absl::StatusCode> new_code,
                                     std::optional<SourceLocation> loc) {
  if (source.ok()) return source;
  if (!new_code) new_code = source.code();

  size_t index = 0;
  std::array<std::string_view, 3> to_join = {};
  if (!prefix_message.empty()) {
    to_join[index++] = prefix_message;
  }
  if (!source.message().empty()) {
    to_join[index++] = source.message();
  }

  absl::Status dest(*new_code, (index > 1) ? std::string_view(absl::StrJoin(
                                                 to_join.begin(),
                                                 to_join.begin() + index, ": "))
                                           : to_join[0]);

  // Preserve the payloads.
  source.ForEachPayload([&](auto name, const absl::Cord& value) {
    dest.SetPayload(name, value);
  });
  if (loc) {
    MaybeAddSourceLocation(dest, *loc);
  }
  return dest;
}

[[noreturn]] void FatalStatus(const char* message, const absl::Status& status,
                              SourceLocation loc) {
  std::fprintf(stderr, "%s:%d: %s: %s\n", loc.file_name(), loc.line(), message,
               status.ToString().c_str());
  std::terminate();
}

}  // namespace internal

std::optional<std::string> AddStatusPayload(absl::Status& status,
                                            std::string_view prefix,
                                            absl::Cord value) {
  std::string payload_id(prefix);
  int i = 1;
  while (true) {
    auto p = status.GetPayload(payload_id);
    if (!p.has_value()) {
      break;
    }
    if (p.value() == value) return std::nullopt;
    payload_id = absl::StrFormat("%s[%d]", prefix, i++);
  }
  status.SetPayload(payload_id, std::move(value));
  return payload_id;
}

}  // namespace tensorstore
