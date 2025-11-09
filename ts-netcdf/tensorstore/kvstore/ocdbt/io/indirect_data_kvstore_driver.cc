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

#include "tensorstore/kvstore/ocdbt/io/indirect_data_kvstore_driver.h"

#include <stdint.h>

#include <cassert>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_ocdbt {
namespace {

ABSL_CONST_INIT internal_log::VerboseFlag ocdbt_logging("ocdbt");

class IndirectDataKvStoreDriver : public kvstore::Driver {
 public:
  explicit IndirectDataKvStoreDriver(kvstore::KvStore base)
      : base_(std::move(base)) {}

  Future<ReadResult> Read(Key key, ReadOptions options) override {
    IndirectDataReference ref;
    ABSL_CHECK(ref.DecodeCacheKey(key));
    TENSORSTORE_ASSIGN_OR_RETURN(auto byte_range,
                                 options.byte_range.Validate(ref.length));
    options.byte_range.inclusive_min = byte_range.inclusive_min + ref.offset;
    // Note: No need to check for overflow in computing `exclusive_max` because
    // `offset` and `length` are validated by `IndirectDataReference::Validate`
    // when the `IndirectDataReference` is decoded.
    options.byte_range.exclusive_max = byte_range.exclusive_max + ref.offset;
    ABSL_LOG_IF(INFO, ocdbt_logging)
        << "read: " << ref << " " << options.byte_range;

    return kvstore::Read(base_, ref.file_id.FullPath(), std::move(options));
  }

  std::string DescribeKey(std::string_view key) override {
    IndirectDataReference ref;
    ABSL_CHECK(ref.DecodeCacheKey(key));
    return tensorstore::StrCat(
        "Byte range ",
        ByteRange{static_cast<int64_t>(ref.offset),
                  static_cast<int64_t>(ref.offset + ref.length)},
        " of ",
        base_.driver->DescribeKey(
            tensorstore::StrCat(base_.path, ref.file_id.FullPath())));
  }

  void GarbageCollectionVisit(
      garbage_collection::GarbageCollectionVisitor& visitor) const final {
    // No-op
  }

  kvstore::KvStore base_;
};

}  // namespace

kvstore::DriverPtr MakeIndirectDataKvStoreDriver(kvstore::KvStore base) {
  return internal::MakeIntrusivePtr<IndirectDataKvStoreDriver>(std::move(base));
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
