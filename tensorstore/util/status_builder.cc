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

#include "tensorstore/util/status_builder.h"

#include <array>
#include <cassert>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/util/status_impl.h"

namespace tensorstore {
namespace internal {

StatusBuilder& StatusBuilder::AddStatusPayload(std::string_view type_url,
                                               absl::Cord payload) {
  auto p = status_.GetPayload(type_url);
  if (!p.has_value()) {
    status_.SetPayload(type_url, std::move(payload));
    return *this;
  }
  if (p.value() == payload) {
    return *this;
  }

  int i = 1;
  while (true) {
    auto payload_id = absl::StrFormat("%s[%d]", type_url, i++);
    if (!status_.GetPayload(payload_id).has_value()) {
      status_.SetPayload(payload_id, std::move(payload));
      return *this;
    }
  }
}

absl::Status StatusBuilder::BuildStatusImpl() const {
  assert(do_build_status());

  const auto& message = rep_.message;
  const bool append = rep_.append;
  std::string buffer;
  std::string_view composed_message =
      [&](const absl::Status& status) -> std::string_view {
    if (status.message().empty()) {
      return message;
    }
    if (message.empty()) {
      return status.message();
    }
    std::array<std::string_view, 3> to_join = {};
    to_join[append ? 0 : 2] = status.message();
    to_join[append ? 2 : 0] = message;
    to_join[1] = ": ";
    buffer = absl::StrJoin(to_join, "");
    return buffer;
  }(status_);

  // Create a new status with the correct message, etc.
  absl::Status dest = internal_status::StatusWithSourceLocation(
      rep_.code, composed_message, loc_);
  internal_status::CopyPayloadsImpl(dest, status_, loc_);
  return dest;
}

}  // namespace internal
}  // namespace tensorstore
