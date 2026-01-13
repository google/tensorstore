// Copyright 2026 The TensorStore Authors
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

#include <stddef.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "tensorstore/internal/source_location.h"

namespace tensorstore {
namespace internal_status {
namespace {
constexpr const char kSourceLocationKey[] = "source locations";

using LineT = decltype(std::declval<SourceLocation>().line());

std::string_view SourceLocFilename(SourceLocation loc) {
  std::string_view filename(loc.file_name());
  if (auto idx = filename.find("tensorstore"); idx != std::string::npos) {
    filename.remove_prefix(idx);
  } else if (auto idx = filename.find("external"); idx != std::string::npos) {
    filename.remove_prefix(idx);
  }
  return filename;
}

std::pair<std::string_view, LineT> LastSourceLocFrom(std::string_view payload) {
  size_t last_newline = payload.rfind('\n');
  std::string_view last_loc_str = last_newline == std::string::npos
                                      ? payload
                                      : payload.substr(last_newline + 1);
  std::pair<std::string_view, std::string_view> file_and_line =
      absl::StrSplit(last_loc_str, absl::MaxSplits(':', 1));

  // Within the same file.
  LineT line = 0;
  if (absl::SimpleAtoi(file_and_line.second, &line)) {
    return std::make_pair(file_and_line.first, line);
  }
  return std::make_pair(file_and_line.first, std::numeric_limits<LineT>::max());
}

}  // namespace

/// Add a source location to the status.
void MaybeAddSourceLocationImpl(absl::Status& status, SourceLocation loc) {
#if TENSORSTORE_HAVE_SOURCE_LOCATION_CURRENT
  if (status.ok()) return;
  if (loc.line() <= 1) return;
  auto payload = status.GetPayload(kSourceLocationKey);

  auto current_loc_file = SourceLocFilename(loc);
  if (!payload.has_value()) {
    status.SetPayload(
        kSourceLocationKey,
        absl::Cord(absl::StrFormat("%s:%d", current_loc_file, loc.line())));
    return;
  }

  // Get the source location from the payload.
  auto last_loc = LastSourceLocFrom(payload->Flatten());
  if (last_loc.first == current_loc_file &&
      std::max(last_loc.second, loc.line()) -
              std::min(last_loc.second, loc.line()) <
          4) {
    // Within 4 lines of the last source location, so we don't need to
    // add this one.
    return;
  }

  payload->Append(absl::StrFormat("\n%s:%d", current_loc_file, loc.line()));
  status.SetPayload(kSourceLocationKey, *std::move(payload));
#endif
}

void CopyPayloadsImpl(absl::Status& dest, const absl::Status& source,
                      SourceLocation loc) {
#if TENSORSTORE_HAVE_SOURCE_LOCATION_CURRENT
  auto source_loc = source.GetPayload(kSourceLocationKey);
  if (source_loc) {
    absl::Cord new_payload = *std::move(source_loc);
    if (auto dest_loc = dest.GetPayload(kSourceLocationKey); dest_loc) {
      new_payload.Append(*dest_loc);
    }
    dest.SetPayload(kSourceLocationKey, std::move(new_payload));
  }
#endif

  // "source location" should have been copied via payload copy
  // Preserve the payloads.
  source.ForEachPayload([&](auto name, const absl::Cord& value) {
#if TENSORSTORE_HAVE_SOURCE_LOCATION_CURRENT
    if (name == kSourceLocationKey) return;
#endif
    dest.SetPayload(name, value);
  });
}

absl::Status StatusWithSourceLocation(absl::StatusCode code,
                                      std::string_view message,
                                      SourceLocation loc) {
  absl::Status status(code, message);
#if TENSORSTORE_HAVE_SOURCE_LOCATION_CURRENT
  if (loc.line() > 1 && code != absl::StatusCode::kOk) {
    status.SetPayload(kSourceLocationKey,
                      absl::Cord(absl::StrFormat(
                          "%s:%d", SourceLocFilename(loc), loc.line())));
  }
#endif
  return status;
}

}  // namespace internal_status
}  // namespace tensorstore
