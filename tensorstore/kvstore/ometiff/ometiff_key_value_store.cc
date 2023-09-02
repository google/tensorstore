// Copyright 2023 The TensorStore Authors
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

#include "tensorstore/internal/kvs_read_streambuf.h"
#include "tensorstore/kvstore/ometiff/ometiff_spec.h"
#include "tensorstore/kvstore/registry.h"

namespace tensorstore {
namespace ometiff {
namespace {

class OMETiffMetadataKeyValueStore : public kvstore::Driver {
 public:
  explicit OMETiffMetadataKeyValueStore(kvstore::DriverPtr base,
                                        std::string key_prefix)
      : base_(std::move(base)), key_prefix_(key_prefix) {}

  Future<ReadResult> Read(Key key, ReadOptions options) override {
    ReadResult result;
    if (options.byte_range != OptionalByteRangeRequest()) {
      // Metadata doesn't need byte range request.
      return absl::InvalidArgumentError("Byte ranges not supported");
    }

    // TODO: plumb in buffer size.
    auto streambuf = internal::KvsReadStreambuf(base_, key, 3 * 1024);
    std::istream stream(&streambuf);
    TENSORSTORE_ASSIGN_OR_RETURN(auto image_info, GetOMETiffImageInfo(stream));
    result.value = absl::Cord(image_info.dump());
    return result;
  }

  void GarbageCollectionVisit(
      garbage_collection::GarbageCollectionVisitor& visitor) const final {
    // No-op
  }

  kvstore::Driver* base() { return base_.get(); }

 private:
  kvstore::DriverPtr base_;
  std::string key_prefix_;
};

}  // namespace
kvstore::DriverPtr GetOMETiffKeyValueStore(kvstore::DriverPtr base_kvstore,
                                           std::string key_prefix) {
  return kvstore::DriverPtr(new OMETiffMetadataKeyValueStore(
      std::move(base_kvstore), std::move(key_prefix)));
}

}  // namespace ometiff
}  // namespace tensorstore