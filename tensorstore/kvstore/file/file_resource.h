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

#ifndef TENSORSTORE_KVSTORE_FILE_FILE_RESOURCE_H_
#define TENSORSTORE_KVSTORE_FILE_FILE_RESOURCE_H_

#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_file_kvstore {

/// When set, the "file" kvstore ensures durability for local file writes
/// (e.g. by calling ::fsync).
///
/// In cases where durability is not required, setting this to ``false`` may
/// make write operations faster.
struct FileIoSyncResource
    : public internal::ContextResourceTraits<FileIoSyncResource> {
  constexpr static bool config_only = true;
  static constexpr char id[] = "file_io_sync";
  using Spec = bool;
  using Resource = Spec;
  static Spec Default() { return true; }
  static constexpr auto JsonBinder() {
    return internal_json_binding::DefaultBinder<>;
  }
  static Result<Resource> Create(
      Spec v, internal::ContextResourceCreationContext context) {
    return v;
  }
  static Spec GetSpec(Resource v, const internal::ContextSpecBuilder& builder) {
    return v;
  }
};

}  // namespace internal_file_kvstore
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_FILE_FILE_RESOURCE_H_
