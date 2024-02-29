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

#ifndef TENSORSTORE_KVSTORE_OCDBT_DUMP_UTIL_H_
#define TENSORSTORE_KVSTORE_OCDBT_DUMP_UTIL_H_

#include <optional>
#include <variant>

#include "absl/strings/cord.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/ocdbt/format/dump.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_ocdbt {

/// Reads and returns the dumped representation of the specified manifest,
/// b+tree node, version tree node, or out-of-line value.
///
/// \param base Specifies base storage of OCDBT database.  Must specify a
///     directory, but a "/" will be appended to `base.path` automatically if
///     not already present.
/// \param node_identifier If `std::nullopt`, dumps the manifest.  Otherwise,
///     dumps the specified node or value.  If `node_identifier` specifies a
///     value, the result is of type `absl::Cord`.  Otherwise the result is of
///     type `::nlohmann::json`.
/// \param context Specifies the context from which the `data_copy_concurrency`
///     and `cache_pool` resources will be used.  If not specified,
///     `Context::Default()` is used.
Future<std::variant<absl::Cord, ::nlohmann::json>> ReadAndDump(
    KvStore base,
    std::optional<LabeledIndirectDataReference> node_identifier = {},
    Context context = {});

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_DUMP_UTIL_H_
