// Copyright 2024 The TensorStore Authors
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

#include "tensorstore/internal/grpc/clientauth/authentication_strategy.h"

#include <optional>
#include <string>
#include <utility>

#include "absl/log/absl_log.h"
#include "tensorstore/internal/os/file_util.h"

namespace tensorstore {
namespace internal_grpc {

std::optional<std::string> LoadCAInfo(const std::string& ca_root_path) {
  if (ca_root_path.empty()) return std::nullopt;
  auto result = internal_os::ReadAllToString(ca_root_path);
  if (!result.ok()) {
    ABSL_LOG(WARNING) << "Failed to read CA root file: " << result.status();
    return std::nullopt;
  }
  return std::move(result).value();
}

}  // namespace internal_grpc
}  // namespace tensorstore
