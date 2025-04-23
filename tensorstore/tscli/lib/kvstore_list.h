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

#ifndef TENSORSTORE_TSCLI_LIB_KVSTORE_LIST_H_
#define TENSORSTORE_TSCLI_LIB_KVSTORE_LIST_H_

#include <stddef.h>

#include <ostream>
#include <string>
#include <string_view>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "tensorstore/context.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace cli {

absl::Status KvstoreList(Context context,
                         tensorstore::kvstore::Spec source_spec,
                         tensorstore::span<std::string_view> match_args,
                         absl::FunctionRef<std::string(
                             size_t, const tensorstore::kvstore::ListEntry&)>
                             formatter,
                         std::ostream& output);

}  // namespace cli
}  // namespace tensorstore

#endif  // TENSORSTORE_TSCLI_LIB_KVSTORE_LIST_H_
