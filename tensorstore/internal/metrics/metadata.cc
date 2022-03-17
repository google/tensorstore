// Copyright 2022 The TensorStore Authors
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

#include "tensorstore/internal/metrics/metadata.h"

#include <string_view>

#include "absl/strings/ascii.h"

namespace tensorstore {
namespace internal_metrics {

// This differs from prometheus; we use path-style metrics.
// https://prometheus.io/docs/instrumenting/writing_clientlibs/
// However, for prometheus export, it should be sufficient to replace all
// '/' with '_'.
bool IsValidMetricName(std::string_view name) {
  if (name.size() < 2) return false;
  if (name[0] != '/') return false;
  if (name[name.size() - 1] == '/') return false;
  if (!absl::ascii_isalpha(name[1])) return false;

  size_t last_slash = 0;
  for (size_t i = 1; i < name.size(); i++) {
    const auto ch = name[i];
    if (ch == '/') {
      if (i - last_slash == 1) return false;
      if (i - last_slash > 63) return false;
      last_slash = i;
    } else if (ch != '_' && !absl::ascii_isalnum(ch)) {
      return false;
    }
  }
  return true;
}

bool IsValidMetricLabel(std::string_view name) {
  if (name.empty()) return false;
  if (!absl::ascii_isalpha(name[0])) return false;
  for (auto ch : name) {
    if (ch != '_' && !absl::ascii_isalnum(ch)) {
      return false;
    }
  }
  return true;
}

}  // namespace internal_metrics
}  // namespace tensorstore
