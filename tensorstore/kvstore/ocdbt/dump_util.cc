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

#include "tensorstore/kvstore/ocdbt/dump_util.h"

#include <optional>
#include <utility>
#include <variant>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/clock.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/internal/cache/cache_pool_resource.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/ocdbt/config.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/dump.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/ocdbt/format/manifest.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/ocdbt/io/indirect_data_kvstore_driver.h"
#include "tensorstore/kvstore/ocdbt/io/io_handle_impl.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_ocdbt {

Future<std::variant<absl::Cord, ::nlohmann::json>> ReadAndDump(
    KvStore base, std::optional<LabeledIndirectDataReference> node_identifier,
    Context context) {
  internal::EnsureDirectoryPath(base.path);
  if (!context) context = Context::Default();
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto data_copy_concurrency_resource,
      context.GetResource<internal::DataCopyConcurrencyResource>());
  if (!node_identifier) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto cache_pool_resource,
        context.GetResource<internal::CachePoolResource>());
    auto io_handle = internal_ocdbt::MakeIoHandle(
        data_copy_concurrency_resource, cache_pool_resource->get(), base,
        /*config_state=*/
        internal::MakeIntrusivePtr<ConfigState>());

    return MapFutureValue(
        data_copy_concurrency_resource->executor,
        [](const ManifestWithTime& manifest_with_time)
            -> Result<std::variant<absl::Cord, ::nlohmann::json>> {
          if (!manifest_with_time.manifest) {
            return absl::NotFoundError("Manifest not found");
          }
          return Dump(*manifest_with_time.manifest);
        },
        io_handle->GetManifest(absl::Now()));
  }

  // Validate the node type.
  if (node_identifier->label != "btreenode" &&
      node_identifier->label != "versionnode" &&
      node_identifier->label != "value") {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Invalid node type: ",
                            tensorstore::QuoteString(node_identifier->label)));
  }

  auto indirect_kvs = MakeIndirectDataKvStoreDriver(base);

  auto key = node_identifier->location.EncodeCacheKey();

  return MapFutureValue(
      data_copy_concurrency_resource->executor,
      [node_identifier, indirect_kvs,
       key = std::move(key)](const kvstore::ReadResult& read_result)
          -> Result<std::variant<absl::Cord, ::nlohmann::json>> {
        if (!read_result.has_value()) {
          return indirect_kvs->AnnotateError(key, "reading",
                                             absl::NotFoundError(""));
        }
        auto& encoded = read_result.value;
        if (node_identifier->label == "value") {
          return encoded;
        } else if (node_identifier->label == "btreenode") {
          TENSORSTORE_ASSIGN_OR_RETURN(
              auto node,
              DecodeBtreeNode(encoded,
                              node_identifier->location.file_id.base_path));
          return Dump(node);
        } else {
          TENSORSTORE_ASSIGN_OR_RETURN(
              auto node,
              DecodeVersionTreeNode(
                  encoded, node_identifier->location.file_id.base_path));
          return Dump(node);
        }
      },
      kvstore::Read(indirect_kvs, node_identifier->location.EncodeCacheKey()));
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
