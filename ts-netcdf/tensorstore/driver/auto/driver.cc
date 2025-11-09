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

#include "tensorstore/driver/driver.h"

#include <cassert>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "tensorstore/context.h"
#include "tensorstore/driver/driver_spec.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/driver/url_registry.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/driver_kind_registry.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/uri_utils.h"
#include "tensorstore/kvstore/auto_detect.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/url_registry.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/open_options.h"
#include "tensorstore/spec.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_auto_detect {

namespace {

namespace jb = ::tensorstore::internal_json_binding;

class AutoDriverSpec
    : public internal::RegisteredDriverSpec<AutoDriverSpec,
                                            /*Parent=*/internal::DriverSpec> {
 public:
  constexpr static char id[] = "auto";
  using Base = internal::RegisteredDriverSpec<AutoDriverSpec,
                                              /*Parent=*/internal::DriverSpec>;

  kvstore::Spec store;
  internal::AllContextResources all_context_resources;

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(internal::BaseCast<internal::DriverSpec>(x), x.store,
             x.all_context_resources);
  };

  OpenMode open_mode() const override { return OpenMode::open; }

  absl::Status ApplyOptions(SpecOptions&& options) override {
    if (options.open_mode != OpenMode::unknown &&
        !(options.open_mode & OpenMode::open)) {
      return absl::InvalidArgumentError(
          "create mode not compatible with format auto-detection");
    }
    if (options.kvstore.valid()) {
      if (store.valid()) {
        return absl::InvalidArgumentError("\"kvstore\" is already specified");
      }
      store = std::move(options.kvstore);
    }
    // TODO(jbms): store staleness bound options
    TENSORSTORE_RETURN_IF_ERROR(schema.Set(static_cast<Schema&&>(options)));
    return absl::OkStatus();
  }

  constexpr static auto default_json_binder = jb::Object(
      jb::Projection<&AutoDriverSpec::store>(jb::KvStoreSpecAndPathJsonBinder),
      jb::Projection<&AutoDriverSpec::all_context_resources>());

  kvstore::Spec GetKvstore() const override { return store; }

  Result<std::string> ToUrl() const override {
    TENSORSTORE_ASSIGN_OR_RETURN(auto base_url, store.ToUrl());
    return tensorstore::StrCat(base_url, "|", id, ":");
  }

  Future<internal::Driver::Handle> Open(
      internal::DriverOpenRequest request) const override;
};

// Implements format auto-detection asynchronously.
//
// The input is an open KvStore, `store`.  The output is an open TensorStore
// Driver handle.
//
// 1. Call `internal_kvstore::AutoDetectFormat` on `store` to obtain
//    auto-detected format candidates.
//
// 2. If there is not exactly one candidate, fail with an error.
//
// 3. If the candidate is a KvStore adapter format, apply it to
//    `store` (wrapped as a `kvstore::Spec`) to obtain a
//    `kvstore::Spec` for the new adapter, then open that adapter
//    using `context` and `store.transaction`.  Then continue at step
//    1, using the new adapter as `store`.
//
// 4. Otherwise, the candidate is a TensorStore KvStore adapter
//    format.  Apply the adapter to `store` (wrapped as a
//    `kvstore::Spec`) to obtain a TensorStore `Spec` for the new
//    format.  Then open that spec using `context`, `spec_options`,
//    and `driver_open_request` (which includes the same transaction
//    as `store.transaction`).
struct AutoOpenState {
  SpecOptions spec_options;
  KvStore store;
  Executor executor;
  Context context;
  internal::DriverOpenRequest driver_open_request;
  using Ptr = std::unique_ptr<AutoOpenState>;
  using PromiseType = Promise<internal::Driver::Handle>;

  static void ContinueAutoDetectWhenReady(Ptr self, PromiseType promise,
                                          Future<KvStore> future) {
    auto& self_ref = *self;
    LinkValue(WithExecutor(
                  self_ref.executor,
                  [self = std::move(self)](PromiseType promise,
                                           ReadyFuture<KvStore> store) mutable {
                    self->store = std::move(store.value());
                    AutoDetect(std::move(self), std::move(promise));
                  }),
              std::move(promise), std::move(future));
  }

  static void AutoDetect(Ptr self, PromiseType promise) {
    auto& self_ref = *self;
    LinkValue(
        WithExecutor(
            self_ref.executor,
            [self = std::move(self)](
                PromiseType promise,
                ReadyFuture<std::vector<internal_kvstore::AutoDetectMatch>>
                    matches_future) mutable {
              auto& matches = matches_future.value();
              if (matches.empty()) {
                promise.SetResult(
                    absl::FailedPreconditionError(tensorstore::StrCat(
                        "Failed to detect format for ",
                        self->store.driver->DescribeKey(self->store.path))));
                return;
              }
              if (matches.size() != 1) {
                promise.SetResult(
                    absl::FailedPreconditionError(tensorstore::StrCat(
                        "Multiple possible formats detected for ",
                        self->store.driver->DescribeKey(self->store.path), ": ",
                        absl::StrJoin(matches, ", "))));
                return;
              }
              ApplyDetectedMatch(std::move(self), std::move(promise),
                                 matches[0]);
            }),
        std::move(promise),
        internal_kvstore::AutoDetectFormat(self_ref.executor, self_ref.store));
  }

  static void ApplyDetectedMatch(
      Ptr self, PromiseType promise,
      const internal_kvstore::AutoDetectMatch& match) {
    auto scheme_kind = internal::GetUrlSchemeKind(match.scheme).value();
    kvstore::Spec wrapped_base;
    wrapped_base.driver =
        internal_kvstore::WrapDriverAsDriverSpec(self->store.driver);
    wrapped_base.path = self->store.path;
    auto& self_ref = *self;
    switch (scheme_kind) {
      case internal::UrlSchemeKind::kKvStoreAdapter: {
        TENSORSTORE_ASSIGN_OR_RETURN(
            auto adapted_spec,
            internal_kvstore::GetAdapterSpecFromUrl(match.scheme,
                                                    std::move(wrapped_base)),
            static_cast<void>(promise.SetResult(std::move(_))));
        ContinueAutoDetectWhenReady(
            std::move(self), std::move(promise),
            kvstore::Open(std::move(adapted_spec), self_ref.context,
                          self_ref.store.transaction));
        return;
      }
      case internal::UrlSchemeKind::kTensorStoreKvStoreAdapter: {
        TENSORSTORE_ASSIGN_OR_RETURN(
            auto spec,
            internal::GetTransformedDriverKvStoreAdapterSpecFromUrl(
                match.scheme, std::move(wrapped_base)),
            static_cast<void>(promise.SetResult(std::move(_))));
        TENSORSTORE_RETURN_IF_ERROR(
            internal::TransformAndApplyOptions(
                spec, std::move(self_ref.spec_options)),
            static_cast<void>(promise.SetResult(std::move(_))));
        TENSORSTORE_RETURN_IF_ERROR(
            DriverSpecBindContext(spec.driver_spec, self_ref.context),
            static_cast<void>(promise.SetResult(std::move(_))));
        LinkResult(std::move(promise),
                   internal::OpenDriver(std::move(spec),
                                        std::move(self->driver_open_request)));
        return;
      }
      default:
        ABSL_LOG(FATAL) << "Unexpected " << scheme_kind << " URL scheme "
                        << match << " auto-detected";
    }
  }
};

Future<internal::Driver::Handle> AutoDriverSpec::Open(
    internal::DriverOpenRequest request) const {
  if (!store.valid()) {
    return absl::InvalidArgumentError("\"kvstore\" must be specified");
  }
  auto [promise, future] = PromiseFuturePair<internal::Driver::Handle>::Make();
  auto state = std::make_unique<AutoOpenState>();
  static_cast<Schema&>(state->spec_options) = schema;
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto data_copy_concurrency,
      all_context_resources.context
          .GetResource<internal::DataCopyConcurrencyResource>());
  state->executor = data_copy_concurrency->executor;
  state->context = all_context_resources.context;
  state->driver_open_request = std::move(request);

  auto kvstore_future =
      kvstore::Open(store, internal::TransactionState::ToTransaction(
                               state->driver_open_request.transaction));
  AutoOpenState::ContinueAutoDetectWhenReady(
      std::move(state), std::move(promise), std::move(kvstore_future));
  return std::move(future);
}

internal::TransformedDriverSpec MakeAutoSpec(kvstore::Spec&& base) {
  auto driver_spec = internal::MakeIntrusivePtr<AutoDriverSpec>();
  driver_spec->store = std::move(base);
  return internal::TransformedDriverSpec{std::move(driver_spec)};
}

Result<internal::TransformedDriverSpec> ParseAutoUrl(std::string_view url,
                                                     kvstore::Spec&& base) {
  auto parsed = internal::ParseGenericUri(url);
  TENSORSTORE_RETURN_IF_ERROR(
      internal::EnsureSchema(parsed, AutoDriverSpec::id));
  TENSORSTORE_RETURN_IF_ERROR(internal::EnsureNoPathOrQueryOrFragment(parsed));
  return MakeAutoSpec(std::move(base));
}

const internal::DriverRegistration<AutoDriverSpec> driver_registration;

const internal::UrlSchemeRegistration url_scheme_registration(
    AutoDriverSpec::id, ParseAutoUrl);

}  // namespace
}  // namespace internal_auto_detect

// Prototype is in tensorstore/auto.h
Spec AutoSpec(kvstore::Spec base) {
  Spec spec;
  using internal_spec::SpecAccess;
  Spec auto_spec;
  SpecAccess::impl(auto_spec) =
      internal_auto_detect::MakeAutoSpec(std::move(base));
  return auto_spec;
}

}  // namespace tensorstore
