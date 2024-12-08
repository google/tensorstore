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

#include "tensorstore/kvstore/gcs_grpc/use_directpath.h"

#include <optional>
#include <string_view>

#include "absl/flags/flag.h"
#include "absl/strings/ascii.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/os/get_bios_info.h"
#include "tensorstore/util/result.h"

ABSL_FLAG(std::optional<bool>, tensorstore_disable_direct_path, std::nullopt,
          "Controls whether to automatically use directpath endpoints in gRPC "
          "to connect to google cloud storage. When set to true, directpath "
          "is not used by default on GCE VMs. "
          "Overrides GOOGLE_CLOUD_DISABLE_DIRECT_PATH.");

namespace tensorstore {
namespace internal_gcs_grpc {
namespace {

bool UseDirectPathGcsEndpointByDefaultImpl() {
  // If the GOOGLE_CLOUD_DISABLE_DIRECT_PATH env var is set then don't use
  // directpath.
  if (auto disable_direct_path =
          internal::GetFlagOrEnvValue(FLAGS_tensorstore_disable_direct_path,
                                      "GOOGLE_CLOUD_DISABLE_DIRECT_PATH");
      disable_direct_path.has_value() && disable_direct_path) {
    return false;
  }

  auto product_name = internal_os::GetGcpProductName();
  if (!product_name.ok()) {
    return false;
  }
  std::string_view product_name_str = product_name.value();
  product_name_str = absl::StripAsciiWhitespace(product_name_str);

  // If the machine is not a GCP machine default to false.
  if (product_name_str != "Google" &&
      product_name_str != "Google Compute Engine") {
    return false;
  }

  // directpath is generally available on GCP machines, so default to true.
  return true;
}

}  // namespace

bool UseDirectPathGcsEndpointByDefault() {
  static bool cached_use_direct_path = UseDirectPathGcsEndpointByDefaultImpl();
  return cached_use_direct_path;
}

}  // namespace internal_gcs_grpc
}  // namespace tensorstore
