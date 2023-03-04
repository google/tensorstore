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

#include "tensorstore/kvstore/ocdbt/test_util.h"

#include <memory>

#include "absl/time/time.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/kvstore/ocdbt/driver.h"
#include "tensorstore/kvstore/ocdbt/format/manifest.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_ocdbt {

Result<std::shared_ptr<const Manifest>> ReadManifest(OcdbtDriver& driver) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto manifest_with_time,
      driver.io_handle_->GetManifest(absl::InfiniteFuture()).result());
  return manifest_with_time.manifest;
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
