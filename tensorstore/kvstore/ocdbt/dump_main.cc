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

#include <iostream>
#include <optional>
#include <string>
#include <type_traits>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/cache/cache_pool_resource.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/init_tensorstore.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json/pprint_python.h"
#include "tensorstore/internal/json_binding/std_optional.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/dump.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/ocdbt/format/manifest.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/ocdbt/io/indirect_data_kvstore_driver.h"
#include "tensorstore/kvstore/ocdbt/io/io_handle_impl.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/json_absl_flag.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"

ABSL_FLAG(tensorstore::JsonAbslFlag<std::optional<tensorstore::kvstore::Spec>>,
          kvstore, std::nullopt, "Underlying kvstore");
ABSL_FLAG(std::string, node, "", "Node/value to dump");

namespace tensorstore {
namespace internal_ocdbt {

namespace {
void PrintValue(const ::nlohmann::json& value) {
  std::cout << internal_python::PrettyPrintJsonAsPython(value) << std::endl;
}

Result<absl::Cord> ReadKey(kvstore::Driver* driver, std::string key) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto read_result, driver->Read(key).result());
  if (!read_result.has_value()) {
    return driver->AnnotateError(key, "reading", absl::NotFoundError(""));
  }
  return read_result.value;
}

absl::Status RunDumpCommand() {
  auto kvs_spec = absl::GetFlag(FLAGS_kvstore).value;
  if (!kvs_spec) {
    return absl::InvalidArgumentError("Must specify --kvstore");
  }
  internal::EnsureDirectoryPath(kvs_spec->path);
  TENSORSTORE_ASSIGN_OR_RETURN(auto kvs, kvstore::Open(*kvs_spec).result());

  std::string node_identifier = absl::GetFlag(FLAGS_node);
  if (node_identifier.empty()) {
    // Print manifest.
    auto context = Context::Default();

    auto io_handle = internal_ocdbt::MakeIoHandle(
        context.GetResource<internal::DataCopyConcurrencyResource>().value(),
        **context.GetResource<internal::CachePoolResource>().value(), kvs,
        /*config_state=*/
        internal::MakeIntrusivePtr<ConfigState>());

    TENSORSTORE_ASSIGN_OR_RETURN(auto manifest_with_time,
                                 io_handle->GetManifest(absl::Now()).result());

    if (!manifest_with_time.manifest) {
      return absl::NotFoundError("Manifest not found");
    }
    PrintValue(Dump(*manifest_with_time.manifest));
    return absl::OkStatus();
  }

  TENSORSTORE_ASSIGN_OR_RETURN(
      auto node_ref, LabeledIndirectDataReference::Parse(node_identifier));

  // Validate the node type.
  if (node_ref.label != "btreenode" && node_ref.label != "versionnode" &&
      node_ref.label != "value") {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Invalid node type: ", tensorstore::QuoteString(node_ref.label)));
  }

  auto indirect_kvs = MakeIndirectDataKvStoreDriver(kvs);

  TENSORSTORE_ASSIGN_OR_RETURN(
      auto encoded,
      ReadKey(indirect_kvs.get(), node_ref.location.EncodeCacheKey()));
  if (node_ref.label == "value") {
    std::cout << encoded;
  } else if (node_ref.label == "btreenode") {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto node,
        DecodeBtreeNode(encoded, node_ref.location.file_id.base_path));
    PrintValue(Dump(node));
  } else {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto node,
        DecodeVersionTreeNode(encoded, node_ref.location.file_id.base_path));
    PrintValue(Dump(node));
  }
  return absl::OkStatus();
}
}  // namespace
}  // namespace internal_ocdbt
}  // namespace tensorstore

int main(int argc, char** argv) {
  tensorstore::InitTensorstore(&argc, &argv);
  auto status = tensorstore::internal_ocdbt::RunDumpCommand();
  if (!status.ok()) {
    std::cerr << status << std::endl;

    if (absl::IsInvalidArgument(status)) {
      std::cerr << "Usage: " << argv[0]
                << " --kvstore <kvstore-json-spec> [--node "
                   "<type>:<file-id>:<offset>:<length> ]"
                << "\n";
      std::cerr << R"(
The kvstore must refer to a prefix/directory containing an OCDBT database.

If `--node` is omitted, the manifest is printed to stdout.

Otherwise, the node <type> must be one of `value`, `btreenode`, or
`versionnode`, as specified in a `location` field within the manifest, a
btreenode, or a versionnode, and the corresponding node data is printed to
stdout.

When printing a manifest, a btreenode, or a versionnode, the decoded value is
printed as a Python literal, rather than JSON, to accommodate byte strings (such
as keys), which cannot be directly represented as JSON (as JSON only supports
Unicode strings).

For a node type of `value`, the raw data is just printed directly to stdout.

Example usage:

rm -rf /tmp/source
rm -rf /tmp/ocdbt
mkdir -p /tmp/source
echo "Hello" > /tmp/source/a
echo "World" > /tmp/source/b

bazel run //tensorstore/kvstore:copy -- --source '"file:///tmp/source/"' --target '{"driver":"ocdbt","base":"file:///tmp/ocdbt/"}'

bazel run //tensorstore/kvstore/ocdbt:dump -- --kvstore '"file:///tmp/ocdbt/"'

echo "Test" > /tmp/source/c
bazel run //tensorstore/kvstore:copy -- --source '"file:///tmp/source/c"' --target '{"driver":"ocdbt","base":"file:///tmp/ocdbt/","path":"c"}'

bazel run //tensorstore/kvstore/ocdbt:dump -- --kvstore '"file:///tmp/ocdbt/"'

)";
      std::cerr << std::flush;
    }
    return 1;
  }
  return 0;
}
