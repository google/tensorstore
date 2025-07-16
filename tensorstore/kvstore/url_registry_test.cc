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

#include "tensorstore/kvstore/url_registry.h"

#include <string>
#include <string_view>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

using ::tensorstore::Future;
using ::tensorstore::IsOkAndHolds;
using ::tensorstore::MatchesStatus;
using ::tensorstore::Result;
using ::tensorstore::internal_kvstore::GetAdapterSpecFromUrl;
using ::tensorstore::internal_kvstore::GetRootSpecFromUrl;
using ::tensorstore::kvstore::DriverPtr;
using ::tensorstore::kvstore::DriverSpec;
using ::tensorstore::kvstore::DriverSpecPtr;
using ::tensorstore::kvstore::Spec;
using ::testing::_;

namespace {

struct TestSpec : public DriverSpec {
 public:
  // None of these methods are actually called.
  virtual absl::Status BindContext(const tensorstore::Context& context) {
    return absl::OkStatus();
  }
  virtual void UnbindContext(
      const tensorstore::internal::ContextSpecBuilder& builder) {}

  virtual void StripContext() {}
  virtual absl::Status ApplyOptions(
      tensorstore::kvstore::DriverSpecOptions&& options) {
    return absl::OkStatus();
  }
  virtual void EncodeCacheKey(std::string* out) const { *out = "test_spec"; }

  virtual std::string_view driver_id() const { return "test"; }
  virtual Result<std::string> ToUrl(std::string_view path) const {
    return absl::UnimplementedError("URL representation not supported");
  }
  virtual DriverSpecPtr Clone() const {
    return tensorstore::internal::MakeIntrusivePtr<TestSpec>();
  }
  virtual Future<DriverPtr> DoOpen() const {
    return absl::UnimplementedError("DoOpen not supported");
  }
  virtual Result<Spec> GetBase(std::string_view path) const { return Spec(); }
  virtual void GarbageCollectionVisit(
      tensorstore::garbage_collection::GarbageCollectionVisitor& visitor)
      const {}
};

Result<Spec> RootHandler(std::string_view url) {
  ABSL_LOG(INFO) << "RootHandler: " << url;
  if (absl::StrContains(url, "root_error")) {
    return absl::InvalidArgumentError("roothandler");
  }
  return {std::in_place, tensorstore::internal::MakeIntrusivePtr<TestSpec>()};
}

Result<Spec> AdapterHandler(std::string_view url, Spec base) {
  ABSL_LOG(INFO) << "AdapterHandler: " << url;
  if (absl::StrContains(url, "adapter_error")) {
    return absl::InvalidArgumentError("adapterhandler");
  }
  return base;
}

const tensorstore::internal_kvstore::UrlSchemeRegistration
    root_handler_registration("root_handler", RootHandler);

const tensorstore::internal_kvstore::UrlSchemeRegistration
    adapter_handler_registration("adapter_handler", AdapterHandler);

TEST(UrlRegistryTest, RootHandlerErrors) {
  EXPECT_THAT(GetRootSpecFromUrl(""),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*URL must be non-empty.*"));

  EXPECT_THAT(GetRootSpecFromUrl("no_colon"),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*URL scheme must be specified.*"));

  EXPECT_THAT(GetRootSpecFromUrl("unregistered_scheme://foo"),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*unsupported URL scheme.*"));

  EXPECT_THAT(GetRootSpecFromUrl("root_handler://root_error"),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*Invalid kvstore URL component.*roothandler.*"));
}

TEST(UrlRegistryTest, RootHandlerOk) {
  EXPECT_THAT(GetRootSpecFromUrl("root_handler://bar"), IsOkAndHolds(_));
}

TEST(UrlRegistryTest, AdapterHandlerErrors) {
  EXPECT_THAT(GetAdapterSpecFromUrl("", Spec()),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*URL must be non-empty.*"));

  EXPECT_THAT(GetAdapterSpecFromUrl("unregistered_scheme", Spec()),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*unsupported URL scheme.*"));
  EXPECT_THAT(GetAdapterSpecFromUrl("adapter_handler:", Spec()),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*Invalid kvstore URL component.*"));

  EXPECT_THAT(
      GetAdapterSpecFromUrl(
          "adapter_handler:adapter_error",
          Spec(tensorstore::internal::MakeIntrusivePtr<TestSpec>())),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    ".*Invalid kvstore URL component.*adapterhandler.*"));
}

TEST(UrlRegistryTest, AdapterHandlerOk) {
  EXPECT_THAT(GetAdapterSpecFromUrl(
                  "adapter_handler",
                  Spec(tensorstore::internal::MakeIntrusivePtr<TestSpec>())),
              IsOkAndHolds(_));
  EXPECT_THAT(GetAdapterSpecFromUrl(
                  "adapter_handler:",
                  Spec(tensorstore::internal::MakeIntrusivePtr<TestSpec>())),
              IsOkAndHolds(_));
  EXPECT_THAT(GetAdapterSpecFromUrl(
                  "adapter_handler://stuff",
                  Spec(tensorstore::internal::MakeIntrusivePtr<TestSpec>())),
              IsOkAndHolds(_));
}

TEST(UrlRegistryTest, SpecFromUrlErrors) {
  EXPECT_THAT(Spec::FromUrl(""),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*URL must be non-empty.*"));

  EXPECT_THAT(Spec::FromUrl("root_handler"),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*URL scheme must be specified.*"));

  EXPECT_THAT(Spec::FromUrl("root_handler://root_error"),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*Invalid kvstore URL component.*roothandler.*"));

  EXPECT_THAT(Spec::FromUrl("root_handler:||"),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*URL must be non-empty.*"));

  EXPECT_THAT(
      Spec::FromUrl(
          "root_handler:|adapter_handler|adapter_handler:adapter_error"),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    ".*Invalid kvstore URL component.*adapterhandler.*"));
}

TEST(UrlRegistryTest, SpecFromUrlOk) {
  EXPECT_THAT(Spec::FromUrl("root_handler://foo|adapter_handler"),
              IsOkAndHolds(_));
}

}  // namespace
