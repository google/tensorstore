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

#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/time/time.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/memory/memory_key_value_store.h"
#include "tensorstore/kvstore/ocdbt/config.h"
#include "tensorstore/kvstore/ocdbt/distributed/cooperator.h"
#include "tensorstore/kvstore/ocdbt/distributed/coordinator_server.h"
#include "tensorstore/kvstore/ocdbt/distributed/rpc_security.h"
#include "tensorstore/kvstore/ocdbt/io/io_handle_impl.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::Context;
using ::tensorstore::KvStore;
using ::tensorstore::internal::CachePool;
using ::tensorstore::internal::CachePtr;
using ::tensorstore::internal::MakeIntrusivePtr;
using ::tensorstore::internal_ocdbt::ConfigConstraints;
using ::tensorstore::internal_ocdbt::ConfigState;
using ::tensorstore::internal_ocdbt::IoHandle;
using ::tensorstore::ocdbt::CoordinatorServer;

namespace internal_ocdbt_cooperator = ::tensorstore::internal_ocdbt_cooperator;

class CooperatorServerTest : public ::testing::Test {
 protected:
  tensorstore::KvStore base_kvstore_;
  IoHandle::Ptr io_handle_;
  CoordinatorServer coordinator_server_;
  internal_ocdbt_cooperator::CooperatorPtr cooperator_;

  CooperatorServerTest() {
    auto security =
        ::tensorstore::internal_ocdbt::GetInsecureRpcSecurityMethod();
    auto cache_pool = CachePool::Make({});
    auto data_copy_concurrency =
        Context::Default()
            .GetResource<tensorstore::internal::DataCopyConcurrencyResource>()
            .value();
    base_kvstore_ = tensorstore::GetMemoryKeyValueStore();
    io_handle_ =
        MakeIoHandle(data_copy_concurrency, cache_pool.get(), base_kvstore_,
                     MakeIntrusivePtr<ConfigState>(
                         ConfigConstraints{},
                         base_kvstore_.driver->GetSupportedFeatures({})));

    {
      CoordinatorServer::Options options;
      options.spec.security = security;
      options.spec.bind_addresses.push_back("localhost:0");
      TENSORSTORE_CHECK_OK_AND_ASSIGN(
          coordinator_server_, CoordinatorServer::Start(std::move(options)));
    }

    std::string coordinator_address =
        tensorstore::StrCat("localhost:", coordinator_server_.port());

    {
      internal_ocdbt_cooperator::Options options;
      options.io_handle = io_handle_;
      options.bind_addresses.push_back("localhost:0");
      options.coordinator_address = coordinator_address;
      options.security = security;
      options.lease_duration = absl::Seconds(10);
      TENSORSTORE_CHECK_OK_AND_ASSIGN(
          cooperator_, internal_ocdbt_cooperator::Start(std::move(options)));
    }
  }
};

TEST_F(CooperatorServerTest, Basic) {
  TENSORSTORE_CHECK_OK(internal_ocdbt_cooperator::GetManifestForWriting(
      *cooperator_, absl::InfinitePast()));
}

}  // namespace
